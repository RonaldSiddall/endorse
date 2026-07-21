"""
Micro mesh generation for mechanical upscaling (apps/upscale_m) — built directly on bgem/endorse.

STRUCTURE mirrors apps/chodby_trans/mesh/create_mesh.py function-for-function:
    make_fractures -> make_geometry (≈ make_geometry_box) -> meshing -> make_gmsh
    -> make_heal_mesh -> make_mesh, orchestrated by make_micro_mesh (returns MicroMesh).
`meshing` stays local for the same reason chodby_trans defines its own: the endorse
`mesh_tools.edz_meshing` hard-codes MinimumCirclePoints, while J. Brezina requires the disc
approximation to be controlled from the config (item B.6); the msh2-write + rename follows
edz_meshing (gmsh chooses the format by extension, bgem GmshIO reads v2 only).

Geometry: a single cube of edge L_ext = L_ext_factor * L_inner CENTERED AT THE ORIGIN (bgem
convention: mesh_tools.box_with_sides and UniformBoxPosition both center at 0) carrying the
boundary conditions. The inner averaging cube [-L/2, L/2]^3 is NOT part of the geometry: averaging
volumes are endorse-side windows applied in postprocess (J. Brezina, meeting 10. 7. 2026), so the
mesh does not conform to them. KUBC results are invariant to the coordinate shift (u = E x changes
by a rigid translation only).

PER-FRACTURE FIELDS (the endorse field pipeline of fullscale_transport/macro_flow_model):
the mesh keeps geometry_gmsh's per-fracture regions fam_<family>_<iii> (their index encodes the
fracture order) all the way through — MicroMesh.mesh_file IS the healed mesh, unmodified.
The per-element mechanical aperture cross_section = aperture_per_r * rho(fracture) is computed
via endorse `fracture_map` (element -> fracture map; the region-JOIN side effect of fracture_map
is irrelevant here — element ids/topology are untouched by it, only region TAGS on an in-memory
copy are, and that copy is never written back over the healed mesh) and written by endorse
`fields_file` into a SEPARATE small field file (MicroMesh.cross_section_file), consumed by the
template as `cross_section: !FieldFE {mesh_data_file: <cross_section_file>, ...}` (the idiom of
J. Brezina's flow_micro_tmpl, with mesh_file and the field file deliberately distinct — as
endorse's own macro_conductivity keeps `mesh_file` and `input_fields_file` separate).

Physical regions in the healed mesh (v2 content, Flow123d ready):
    box        (3D)  the whole rock domain
    fam_<family>_<iii>  (2D)  one region PER FRACTURE (bgem geometry_gmsh naming, never
      renamed/joined) — the template's <fracture_regions> placeholder is the full bracket-list
      of these names (Flow123d accepts a region list wherever a single region is accepted)
    .side_[xyz][01]  (2D)  boundary faces for BC prescription — the endorse box_with_sides names
      with the chodby_trans '.'+side convention (all six get the same KUBC formula anyway)
The physical name -> region id map is read back from the healed mesh (bgem GmshIO) and carried in
MicroMesh.regions, replacing the previously hard-coded region tags.

Fracture input (cfg.fractures):
  - mode 'explicit': list of penny-shaped fractures given by GEOMETRIC disc radius r, center and
    normal (centered coordinates). Converted to bgem Fracture objects at this boundary: bgem
    EllipseShape is the disc of UNIT AREA, so bgem size = sqrt(area) = r * sqrt(pi).
  - mode 'stochastic': cfg.fractures.stochastic.population is passed VERBATIM to
    bgem Population.from_cfg (B.6); sampling via endorse `mesh_tools.generate_fractures`
    ('seed', optional 'sample_range', optional 'n_frac_limit' with endorse range adaptation).
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from bgem.gmsh import gmsh, gmsh_io, heal_mesh, options
from bgem.stochastic import EllipseShape, Fracture, FractureSet, Population
from bgem.stochastic import fr_mesh
from bgem.upscale.fem import Grid
from endorse.common import dotdict, File, memoize
from endorse.fullscale_transport import fracture_map
from endorse.macro_flow_model import fields_file
from endorse.mesh import mesh_tools
from endorse.mesh.fracture_tools import RegionFracture
from endorse.mesh_class import load_mesh

SQRT_PI = float(np.sqrt(np.pi))


def _ellipse_gmsh_base_shape(self, gmsh_geom):
    # workaround for bgem@JB_homo: EllipseShape.gmsh_base_shape references the undefined
    # attribute `self.scale` (AttributeError); the unit-area disc radius is `self.R`.
    # Remove once fixed upstream (question recorded in PLAN.md).
    return gmsh_geom.disc(rx=self.R, ry=self.R)


_orig_fracture_set_from_list = FractureSet.from_list.__func__


def _fracture_set_from_list(cls, fr_list):
    # workaround for bgem@JB_homo: FractureSet.from_list (bgem/stochastic/fr_set.py)
    # unconditionally sets family_idx = np.full(len(fr_list), 0), discarding each input
    # Fracture's own `.family` (correctly set per-fracture by Population.sample() for stochastic
    # DFNs). Harmless for 'explicit' mode (Fracture.family is always None there, family is
    # meaningless), but for 'stochastic' mode with 2+ population families this silently collapses
    # every fracture's region into fam_0_<iii> regardless of which population family it actually
    # came from (fr_mesh.geometry_gmsh names regions "fam_{fr.family}_{i:03d}") -- confirmed
    # empirically (R. Siddall report, 2026-07-14): a 2-family, 35-fracture DFN sampled correctly
    # as {family 0: 15, family 1: 20} but read back as {family 0: 35} after from_list. Restore
    # the real per-fracture family here; falls back to from_list's own default (0) whenever
    # every input Fracture.family is None (explicit mode), so that mode is unaffected.
    # Remove once fixed upstream (question recorded in PLAN.md).
    fr_set = _orig_fracture_set_from_list(cls, fr_list)
    families = [getattr(fr, "family", None) for fr in fr_list]
    if any(f is not None for f in families):
        fr_set.family = np.array([0 if f is None else f for f in families], dtype=np.int32)
    return fr_set


@dataclass
class MicroMesh:
    """Result of make_micro_mesh."""
    # the healed mesh, unmodified — the single Flow123d `mesh_file`
    mesh_file: File
    # per-element cross_section (aperture) field, a SEPARATE small file (endorse fields_file);
    # Flow123d reads it via `!FieldFE {mesh_data_file: <cross_section_file>}` — same element ids
    # as mesh_file, so it lines up automatically (see module docstring)
    cross_section_file: File
    fractures: FractureSet
    # inner averaging window bounds: (lo, hi) so the window is [lo, hi]^3
    inner_box: Tuple[float, float]
    # outer cube edge
    L_ext: float
    # physical name -> (region id, dim), read from the healed mesh
    regions: Dict[str, Tuple[int, int]]
    # one region name PER FRACTURE (bgem geometry_gmsh's fam_<family>_<iii>, never joined)
    fracture_regions: List[str]

    @property
    def bulk_region_id(self) -> int:
        return self.regions["box"][0]

    @property
    def fracture_region_ids(self) -> List[int]:
        """Region ids in the mesh's own physical-group order (fracture_regions is NOT sorted) —
        fine for set/membership use (region classification via endorse Subdomain.create, see
        postprocess._window_subdomain), where order is irrelevant; use
        fracture_radius_by_region_id where order-independent correctness matters (e.g. matching a
        region to its radius)."""
        return [self.regions[name][0] for name in self.fracture_regions]

    @property
    def fracture_radii(self) -> np.ndarray:
        """Geometric disc radii rho, in FractureSet GENERATION order; bgem stores size
        r = sqrt(area) = rho * sqrt(pi)"""
        if len(self.fractures) == 0:
            return np.array([])
        return self.fractures.radius[:, 0] * self.fractures.base_shape.R

    @property
    def fracture_radius_by_region_id(self) -> Dict[int, float]:
        """
        region id -> geometric radius, correctly matched via the GENERATION index ENCODED IN
        each region's own name (bgem fr_mesh.geometry_gmsh: f"fam_{fr.family}_{i:03d}", i =
        position in the FractureSet), independent of fracture_regions' own iteration order.
        Naively zipping fracture_region_ids against fracture_radii (generation order) — as an
        earlier version of this code and debug_postprocess.py did, when fracture_regions was
        still alphabetically sorted — silently pairs a fracture with ANOTHER fracture's radius as
        soon as the two orders diverge: zero-padding to 3 digits kept alphabetical == numeric
        order only for indices 0-999, so that broke first at exactly 1000 fractures, not
        gradually — confirmed empirically (chatgpt-codex-connector PR review, 2026-07-14): a
        5615-fracture two-family DFN test mismatched 5514/5615 region-to-radius pairs this way.
        fracture_regions is no longer sorted (J. Brezina PR review), but this method stays: it is
        correct regardless of fracture_regions' order, so it needs no assumption about what order
        the mesh happens to declare physical groups in.
        """
        radii = self.fracture_radii
        return {
            self.regions[name][0]: float(radii[int(name.rsplit("_", 1)[1])])
            for name in self.fracture_regions
        }

    def aperture_by_region_id(self, aperture_per_r: float) -> Dict[int, float]:
        """UNDEFORMED per-fracture aperture, keyed by region id via
        fracture_radius_by_region_id -- single source of truth for kubc.run_kubc and
        debug_postprocess.main, which both need exactly this dict."""
        return {rid: aperture_per_r * radius
                for rid, radius in self.fracture_radius_by_region_id.items()}


def make_averaging_grid(cfg_geometry: dotdict, micro: MicroMesh) -> Grid:
    """
    bgem Grid of averaging windows over the inner cube (NOT imprinted in the mesh) -- single
    source of truth for run_upscale.main, kubc.run_kubc and debug_postprocess.main, which all
    need exactly this grid (geometry.subdomains = [1,1,1] gives the classic single window).
    """
    lo, hi = micro.inner_box
    subdivision = np.asarray(cfg_geometry.get("subdomains", [1, 1, 1]))
    return Grid(np.full(3, hi - lo), subdivision, origin=np.full(3, lo))


@memoize
def make_fractures(cfg_fractures: dotdict, box) -> FractureSet:
    """
    Fracture set for the domain box (centered at origin), per cfg (explicit or stochastic).
    Mirrors chodby_trans create_mesh.make_fractures (memoized; the seed lives in the config).
    Stochastic sampling goes through endorse mesh_tools.generate_fractures (range adaptation
    for optional n_frac_limit, seed handling); the population config is Population.from_cfg
    verbatim
    """
    mode = str(cfg_fractures.mode).strip().lower()
    if mode == "explicit":
        fr_list = []
        for item in cfg_fractures.explicit_list:
            normal = np.asarray(item["normal"], dtype=float)
            fr_list.append(Fracture(EllipseShape.id, float(item["r"]) * SQRT_PI, np.asarray(item["center"], dtype=float),
                                    normal / np.linalg.norm(normal)))
        return FractureSet.from_list(fr_list)
    elif mode == "stochastic":
        cfg_st = cfg_fractures.stochastic
        population = Population.from_cfg(cfg_st.population, box, shape=EllipseShape())
        sample_range = tuple(float(r) for r in cfg_st.sample_range) \
            if cfg_st.get("sample_range", None) is not None else (None, None)
        fractures = mesh_tools.generate_fractures(
            population, sample_range, cfg_st.get("n_frac_limit", None), box,
            int(cfg_st.get("seed", 0)))
        fr_set = FractureSet.from_list(fractures)
        print(f"[upscale_m dfn] sampled {len(fr_set)} fracture(s), bgem sizes in "
              f"[{fr_set.radius[:, 0].min():.4g}, {fr_set.radius[:, 0].max():.4g}]")
        return fr_set
    raise ValueError(f"Unknown fractures.mode: {cfg_fractures.mode!r}")


def _corner_tetrahedron(factory, corner: np.ndarray, d: float):
    """
    Small support tetrahedron for SUBC's rigid-body-mode removal: one vertex AT an outer-cube
    `corner`, the other three arms offset by `d` further OUTWARD (same sign as the corner's own
    coordinate on that axis) along X, Y, Z — same shape as R. Siddall's original static_bc
    pipeline (Python_scripts/.../generate_static_fracture.py:create_occ_tetrahedron), built here
    through factory.model (== gmsh.model.occ, the same primitive calls GeometryOCC.make_simplex
    itself uses) instead of a separate raw-gmsh helper, and generalized to any corner/sign
    instead of one hand-written call per corner.
    """
    sign = np.sign(corner)
    arms = corner + d * sign * np.eye(3)  # arms[0]=X-arm, arms[1]=Y-arm, arms[2]=Z-arm
    p_common = factory.model.addPoint(*corner)
    p_arm = [factory.model.addPoint(*a) for a in arms]

    l_common = [factory.model.addLine(p_common, p_arm[i]) for i in range(3)]
    l_ring = [factory.model.addLine(p_arm[i], p_arm[(i + 1) % 3]) for i in range(3)]

    def face(loop):
        return factory.model.addPlaneSurface([factory.model.addCurveLoop(loop)])

    f_z = face([l_common[0], l_ring[0], -l_common[1]])    # {common, armX, armY}: const Z
    f_x = face([l_common[1], l_ring[1], -l_common[2]])    # {common, armY, armZ}: const X
    f_y = face([l_common[2], l_ring[2], -l_common[0]])    # {common, armZ, armX}: const Y
    f_outer = face([-l_ring[0], -l_ring[2], -l_ring[1]])  # {armX, armY, armZ}: outer, unused

    vol = factory.model.addVolume([factory.model.addSurfaceLoop([f_x, f_y, f_z, f_outer])])
    factory._need_synchronize = True
    return factory.object(3, vol)


def _support_faces(factory, tet, corner: np.ndarray, axis_names: Dict[str, str], tol: float):
    """
    After fragmentation, pick out `tet`'s boundary face(s) whose centroid lies in the coordinate
    plane(s) named by `axis_names` (e.g. {"x": "support_origin_X", ...}) — the SAME axis-aligned
    faces R. Siddall's original pipeline extracted via a bounding-box query
    (generate_static_fracture.py:130-135). Done geometrically here since fragment() invalidates
    any tags recorded before it runs, so pre-fragment face identities can't be tracked directly.
    """
    boundary = tet.get_boundary()
    picked = {}
    for dim, tag in boundary.dim_tags:
        face_obj = factory.object(dim, tag)
        center, mass = face_obj.center_of_mass()
        if mass == 0:
            continue
        for axis, name in axis_names.items():
            if abs(center["xyz".index(axis)] - corner["xyz".index(axis)]) < tol:
                picked[name] = face_obj
    missing = set(axis_names.values()) - picked.keys()
    assert not missing, f"support face(s) {missing} not found at corner {corner}"
    return picked


# The 3 support corners (3-2-1 scheme: 3+2+1 = 6 constraints = the 6 rigid-body modes), on the
# OUTER cube's bottom (min-Z) face, named exactly as R. Siddall's original static_bc pipeline.
def _support_spec(half: float) -> List[Tuple[np.ndarray, Dict[str, str]]]:
    return [
        (np.array([-half, -half, -half]),
         {"x": "support_origin_X", "y": "support_origin_Y", "z": "support_origin_Z"}),
        (np.array([half, -half, -half]),
         {"y": "support_tetra_one_norm_Y", "z": "support_tetra_one_norm_Z"}),
        (np.array([-half, half, -half]),
         {"z": "support_tetra_two_norm_Z"}),
    ]


def make_geometry(factory, cfg_geometry: dotdict, cfg_mesh: dotdict, fracture_set: FractureSet,
                  subc_support: bool = False):
    """
    Outer cube + fractures + named boundary sides (+ SUBC support tetrahedra, if `subc_support`);
    one mutual fragment. Mirrors chodby_trans create_mesh.make_geometry_box (incl. the
    '.'+side_name boundary regions). Fractures KEEP geometry_gmsh's per-fracture regions
    fam_<family>_<iii> — endorse fracture_map joins them after healing and uses them for the
    element -> fracture map.
    """
    L_ext = float(cfg_geometry.get("L_ext_factor", 2.0)) * float(cfg_geometry.L_inner)
    box_dims = [L_ext, L_ext, L_ext]

    box, sides = mesh_tools.box_with_sides(factory, box_dims)  # centered at origin
    geometry_set = [box]
    if len(fracture_set) > 0:
        fr_shapes, _region_map = fr_mesh.geometry_gmsh(fracture_set, factory)
        fr_group = fr_shapes.intersect(box).mesh_step(float(cfg_mesh.fracture_mesh_step))
        geometry_set.append(fr_group)
    else:
        print("[upscale_m mesh] WARNING: empty fracture set, mesh has no fracture regions")
    n_main = len(geometry_set)

    support_spec = _support_spec(L_ext / 2.0) if subc_support else []
    d = float(cfg_geometry.get("support_fraction_d", 0.1)) * float(cfg_geometry.L_inner)
    geometry_set += [_corner_tetrahedron(factory, corner, d) for corner, _ in support_spec]

    factory.synchronize()
    fragmented = factory.fragment(*geometry_set, *sides.values())
    geometry_final = fragmented[:n_main]

    if support_spec:
        support_tets = fragmented[n_main:n_main + len(support_spec)]
        for (corner, axis_names), tet in zip(support_spec, support_tets):
            faces = _support_faces(factory, tet, corner, axis_names, tol=1e-9 * L_ext)
            geometry_final += [f.set_region('.' + name) for name, f in faces.items()]
        geometry_final.append(factory.group(*support_tets).set_region("support_tetras_volume"))

    side_start = n_main + len(support_spec)
    for side_name, side_fr in zip(sides.keys(), fragmented[side_start:]):
        geometry_final.append(side_fr.set_region('.' + side_name))

    geometry_final = factory.group(*geometry_final)
    factory.synchronize()
    factory.keep_only(geometry_final)
    factory.synchronize()
    factory.remove_duplicate_entities()
    factory.synchronize()
    return geometry_final


def meshing(factory, objects, mesh_filename: str, cfg_mesh: dotdict):
    """
    Common meshing setup, mirrors chodby_trans create_mesh.meshing and endorse edz_meshing
    (bgem mesh options; kept local so MinimumCirclePoints stays config-driven, see B.6).
    The mesh is written as v2 content (bgem GmshIO/Flow123d) and renamed to the target name,
    exactly as endorse mesh_tools.edz_meshing does.
    """
    step = float(cfg_mesh.fracture_mesh_step)
    factory.write_brep(os.path.splitext(mesh_filename)[0])
    factory.mesh_options.CharacteristicLengthMin = step / float(cfg_mesh.get("mesh_size_min_fraction", 10.0))
    factory.mesh_options.CharacteristicLengthMax = step
    # disc approximation: minimum number of boundary points per full circle (J. Brezina, B.6)
    factory.mesh_options.MinimumCirclePoints = int(cfg_mesh.get("min_circle_points", 16))
    factory.mesh_options.MinimumCurvePoints = int(cfg_mesh.get("min_curve_points", 16))
    factory.mesh_options.ToleranceInitialDelaunay = float(cfg_mesh.get("tolerance_initial_delaunay", 0.01))
    factory.mesh_options.Algorithm = options.Algorithm3d.Delaunay

    factory.make_mesh(objects, dim=3, eliminate=False)
    factory.write_mesh(format=gmsh.MeshFormat.msh2)  # writes <model_name>.msh2 (v2 content)
    os.rename(factory.model_name + ".msh2", mesh_filename)


def make_gmsh(cfg_geometry: dotdict, cfg_mesh: dotdict, fracture_set: FractureSet,
              work_dir: str = ".", subc_support: bool = False) -> File:
    """
    Geometry + meshing in a fresh gmsh model. Mirrors chodby_trans create_mesh.make_gmsh.
    `subc_support` adds the SUBC support tetrahedra (see make_geometry) — off by default so the
    KUBC-only mesh is completely unaffected.
    """
    mesh_name = cfg_mesh.get("mesh_name", "micro_mesh")
    final_mesh_filename = os.path.join(work_dir, mesh_name + ".msh")

    factory = gmsh.GeometryOCC(mesh_name, verbose=False)
    factory.geom_options.Tolerance = float(cfg_mesh.get("tolerance", 1e-6))
    factory.geom_options.ToleranceBoolean = float(cfg_mesh.get("tolerance_boolean", 1e-5))

    geometry_final = make_geometry(factory, cfg_geometry, cfg_mesh, fracture_set, subc_support)

    print(f"[upscale_m mesh] meshing: L_ext={cfg_geometry.get('L_ext_factor', 2.0)}*"
          f"{cfg_geometry.L_inner}, step={cfg_mesh.fracture_mesh_step}, "
          f"{len(fracture_set)} fracture(s)")
    meshing(factory, [geometry_final], final_mesh_filename, cfg_mesh)

    if str(cfg_mesh.get("show_gui", "no")).lower() in ("yes", "y", "true"):
        try:
            factory.show()
        except Exception as e:
            print(f"[upscale_m mesh] gmsh GUI unavailable: {e}")
    factory.close()  # release gmsh so repeated in-process meshing works
    return File(final_mesh_filename)


def make_heal_mesh(mesh_name: str, mesh_file: File, work_dir: str = ".",
                   gamma_tol: float = 0.01) -> Path:
    """
    Heal the raw mesh (required for stable Flow123d runs). Mirrors chodby_trans
    create_mesh.make_heal_mesh incl. the file-existence guard.
    (The endorse load_mesh(heal_tol) path is NOT used: its heal branch is broken upstream —
    mesh_class.py:73 calls the nonexistent Mesh.load_mesh; question in PLAN.md.)
    """
    mesh_file_healed = Path(work_dir) / (mesh_name + "_healed.msh")
    if not mesh_file_healed.exists():
        print("[upscale_m mesh] healing mesh ...")
        hm = heal_mesh.HealMesh.read_mesh(mesh_file.path)
        hm.heal_mesh(gamma_tol=gamma_tol)
        hm.write(file_name=str(mesh_file_healed))
        print(f"[upscale_m mesh] healed mesh written: {mesh_file_healed}")
    return mesh_file_healed


def make_mesh(cfg_geometry: dotdict, cfg_mesh: dotdict, fracture_set: FractureSet,
              work_dir: str = ".", subc_support: bool = False) -> File:
    """
    Raw mesh (skipped when the file already exists, as chodby_trans create_mesh.make_mesh)
    + healing. Returns the healed mesh File.
    NOTE: the file-existence guard above does not know whether a cached mesh was built with
    `subc_support` — switching loads.bc_type to/from subc on an EXISTING run dir reuses the stale
    mesh regardless (same staleness class as switching fractures.mode, see mesh_name's own
    comment in config.yaml); use a fresh RUN_NAME or mesh_name when doing so.
    """
    mesh_name = cfg_mesh.get("mesh_name", "micro_mesh")
    raw_mesh_path = Path(work_dir) / (mesh_name + ".msh")
    if not raw_mesh_path.exists():
        mesh_file = make_gmsh(cfg_geometry, cfg_mesh, fracture_set, work_dir, subc_support)
    else:
        mesh_file = File(str(raw_mesh_path))

    mesh_file_healed = make_heal_mesh(mesh_name, mesh_file, work_dir,
                                      gamma_tol=float(cfg_mesh.get("heal_gamma_tol", 0.01)))
    return File(str(mesh_file_healed))


def make_cross_section_field(healed_file: File, fracture_set: FractureSet, aperture_per_r: float,
                             field_path: Path) -> None:
    """
    The endorse field pipeline (fullscale_transport/macro_flow_model) for the mechanical
    aperture: load the healed mesh (endorse load_mesh), rebuild the per-fracture RegionFracture
    list from the fam_* physical names (their index = fracture order in the FractureSet) and use
    endorse `fracture_map` for the element -> fracture map (its region-JOIN side effect only
    touches this in-memory copy, never the healed mesh on disk — see module docstring), fill
    cross_section[el] = aperture_per_r * rho(fracture) and write it to a SEPARATE small field
    file via endorse `fields_file` (msh2 content + vtu preview) — analogous to endorse
    macro_flow_model's own `input_fields_file`, distinct from the geometry `mesh_file`.
    (endorse apply_fields.fr_fields_repo is NOT reused: it is transmissivity-specific,
    tr_a/tr_b power law + cubic-law conductivity.)
    """
    mesh = load_mesh(healed_file)
    cross_section = np.ones(len(mesh.elements))
    fam_names = sorted((n for n in mesh.gmsh_io.physical if n.startswith("fam_")),
                       key=lambda n: int(n.rsplit("_", 1)[1]))
    if fam_names:
        region_fractures = [
            RegionFracture(fracture_set[int(n.rsplit("_", 1)[1])], gmsh.Region.get(n))
            for n in fam_names]
        elm_to_ifr = fracture_map(mesh, region_fractures, n_large=0, dim=3)
        # geometric radii of the PRESENT fractures; bgem size r = rho * sqrt(pi)
        rho = np.array([fr.r for fr in region_fractures]) * fracture_set.base_shape.R
        el_idx = np.fromiter(elm_to_ifr.keys(), dtype=int)
        i_fr = np.fromiter(elm_to_ifr.values(), dtype=int)
        cross_section[el_idx] = aperture_per_r * rho[i_fr]
    fields_file(mesh, dict(cross_section=cross_section), file_name=str(field_path))


def make_micro_mesh(cfg_geometry: dotdict, cfg_mesh: dotdict, cfg_fractures: dotdict,
                    aperture_per_r: float, work_dir: str = ".",
                    subc_support: bool = False) -> MicroMesh:
    """
    Fracture set + mesh + healing + the separate cross_section field file (all files written
    into `work_dir`); the region map is read back from the healed mesh itself.
    Top-level equivalent of chodby_trans create_mesh.main.
    `subc_support` (set for loads.bc_type in {subc, both}): adds the support tetrahedra needed to
    remove SUBC's rigid-body modes (make_geometry) — the buffer's Saint-Venant decay does NOT
    substitute for this, see PLAN.md.
    """
    # patch the two real bgem bugs above in place, here rather than at module import time so
    # merely importing micro_mesh has no side effect on bgem process-wide
    EllipseShape.gmsh_base_shape = _ellipse_gmsh_base_shape
    FractureSet.from_list = classmethod(_fracture_set_from_list)

    L = float(cfg_geometry.L_inner)
    factor = float(cfg_geometry.get("L_ext_factor", 2.0))
    assert factor >= 1.0, f"L_ext_factor must be >= 1 (got {factor})"
    L_ext = factor * L

    fracture_set = make_fractures(cfg_fractures, [L_ext, L_ext, L_ext])
    healed_file = make_mesh(cfg_geometry, cfg_mesh, fracture_set, work_dir, subc_support)

    mesh_name = cfg_mesh.get("mesh_name", "micro_mesh")
    cross_section_path = Path(work_dir) / (mesh_name + "_cross_section.msh")
    if not cross_section_path.exists():
        make_cross_section_field(healed_file, fracture_set, float(aperture_per_r), cross_section_path)
        print(f"[upscale_m mesh] cross_section field written: {cross_section_path}")

    # physical name -> (region id, dim), authoritative for postprocess element selection
    regions = dict(gmsh_io.GmshIO(healed_file.path).physical)
    # NOT sorted (J. Brezina PR review): keep the mesh's own physical-group order rather than
    # imposing a lexicographic one (which breaks fam_<family>_<iii> at 1000+ fractures per family,
    # see fracture_radius_by_region_id below).
    fracture_regions = [n for n in regions if n.startswith("fam_")]

    return MicroMesh(
        mesh_file=healed_file,
        cross_section_file=File(str(cross_section_path)),
        fractures=fracture_set,
        inner_box=(-L / 2.0, L / 2.0),
        L_ext=L_ext,
        regions=regions,
        fracture_regions=fracture_regions,
    )
