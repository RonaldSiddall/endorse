"""
Volume integration of stress and strain over averaging windows.

Delegates the geometric work -- candidate search, cut-boundary fractional weight, per-element
measure -- to endorse `homogenisation` (`Subdomain.create` + `MacroCube` + `WindowsMacroMesh`,
J. Brezina PR review, 14.7 + 20.7: `Subdomain.create` used to hard-filter to `el_dim_slice(dim=3)`
and `Mesh.el_volumes` used to return 0.0 for non-tets, both fixed upstream so BULK and FRACTURE
elements are now handled by the SAME endorse mechanism, no local duplicate). `Subdomain.create`
has no concept of "which regions are relevant" or "fracture aperture" -- both stay app-side:
`_window_subdomain` below filters candidates to {bulk_region_id, *fracture_region_ids} and builds
the per-element measure (true volume for bulk, area * aperture thin-plate overlay for fracture,
report sec. 3.1) passed into `Subdomain.weighted_sum`:

  <sigma> = (1/V) sum_e w_e measure_e sigma_e,  measure_e = V_e (bulk) or A_e * delta_e (fracture)
  <eps>   = (1/V) sum_e w_e measure_e eps_e,    eps = M : sigma

V = the DECLARED window volume (bgem Grid.step product), not the captured measure -- the two
agree up to the O(2^-level) cut-boundary error of the weight estimate itself, since a fracture has
zero true 3D volume and bulk tets exactly tile the window (R. Siddall's GeneralComputationClass
_V_RVE(use_actual_volume=False) convention).

The only local physics is `hooke_strain` — the isotropic inverse Hooke law
eps = ((1+nu) sigma - nu tr(sigma) I) / E (report sec. 3.4.2). AUDITED: no isotropic
stiffness/compliance builder exists anywhere in endorse or bgem (see PLAN.md log 2026-07-11).
Fracture elements are thin elastic plates weighted by the aperture delta (report sec. 3.1); the
adequacy of this treatment vs the explicit displacement-jump term J is an open question in PLAN.md.

Voigt convention (report): sigma = [11, 22, 33, 23, 13, 12]; strain shear rows carry engineering
shear gamma = 2 eps.
"""
import os
from typing import Dict, List, Tuple

import numpy as np

from bgem.gmsh.gmsh_io import GmshIO
from bgem.upscale.fem import Grid
from endorse.homogenisation import MacroCube, Subdomain, WindowsMacroMesh
from endorse.mesh_class import Mesh, load_mesh

from micro_mesh import MicroMesh

# engineering shear on the Voigt strain shear rows: gamma = 2 eps (report convention)
ENG_SHEAR = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])


def hooke_strain(sigma: np.ndarray, young: float, poisson: float) -> np.ndarray:
    """
    Isotropic inverse Hooke law, tensor form: eps = ((1+nu) sigma - nu tr(sigma) I) / E.
    Input/output (N, 3, 3). (No such builder exists in endorse/bgem — audited, PLAN.md.)
    """
    trace = np.trace(sigma, axis1=1, axis2=2)
    return ((1.0 + poisson) * sigma - poisson * trace[:, None, None] * np.eye(3)) / young


def _window_subdomain(mesh: Mesh, region_id_field: np.ndarray, bulk_region_id: int,
                      fracture_region_ids: List[int], aperture_by_region_id: Dict[int, float],
                      macro_mesh: WindowsMacroMesh, i_window: int, level: int
                      ) -> Tuple[Subdomain, np.ndarray]:
    """
    Elements + weights + per-element measure for ONE averaging window: endorse Subdomain.create
    (candidate search + MacroCube's fractional cut-boundary weight) filtered down to the bulk/
    fracture regions this app cares about (endorse has no such concept), with measure = true
    volume for bulk, aperture-scaled effective volume for fracture (area * aperture, report sec.
    3.1) -- so Subdomain.weighted_sum sums bulk and fracture contributions with consistent (m^3)
    units. Returns (subdomain, measures) since Subdomain itself stays agnostic to the measure.
    """
    relevant_ids = {bulk_region_id, *fracture_region_ids}
    sub = Subdomain.create(MacroCube(level=level), mesh, macro_mesh, i_window, dims=(2, 3))
    keep = np.isin(region_id_field[sub.el_indices], list(relevant_ids))
    el_indices = np.asarray(sub.el_indices)[keep]
    intersect_weights = np.asarray(sub.intersect_weights)[keep]
    sub = Subdomain(sub.mesh, sub.macro_el_idx, list(el_indices), list(intersect_weights))

    is_bulk = region_id_field[sub.el_indices] == bulk_region_id
    aperture_scale = np.ones(len(sub.el_indices))
    frac_mask = ~is_bulk
    if frac_mask.any():
        aperture_scale[frac_mask] = [aperture_by_region_id[r]
                                     for r in region_id_field[sub.el_indices][frac_mask]]
    measures = mesh.el_volumes[sub.el_indices] * aperture_scale
    return sub, measures


def write_subdomain_meshes(micro: MicroMesh, grid: Grid, level: int, aperture_per_r: float,
                           out_dir: str) -> None:
    """
    Diagnostic export (output.write_subdomain_meshes): one filtered .msh per averaging window,
    containing ONLY the elements _window_subdomain selects for that window (same selection
    average_windows uses, via the shared WindowsMacroMesh bounds) — so a window's effective-tensor
    report can be paired with exactly the elements it was computed from and inspected directly in
    GMSH/ParaView, without re-deriving the selection by hand (as was done ad hoc for
    dfn_test_222_level_3, see PLAN.md 2026-07-17).

    Reads the PRE-SOLVE healed mesh (micro.mesh_file) — geometry only, no Flow123d output needed,
    so this is the SAME regardless of which BC type's results the report is for. Kept as an
    explicit, separate call per BC type at the call site (not shared/deduplicated between kubc/
    and subc/ output) rather than assuming the selection can never differ (R. Siddall, 2026-07-17).
    """
    mesh = load_mesh(micro.mesh_file)
    region_id_field = np.array([el.tags[0] for el in mesh.elements])
    aperture_by_region_id = micro.aperture_by_region_id(aperture_per_r)
    gio = mesh.gmsh_io
    macro_mesh = WindowsMacroMesh(grid)
    n_sub = len(macro_mesh.elements)

    for i_sub in range(n_sub):
        sub, _ = _window_subdomain(mesh, region_id_field, micro.bulk_region_id,
                                   micro.fracture_region_ids, aperture_by_region_id,
                                   macro_mesh, i_sub, level)
        keep_tags = {mesh.el_ids[i] for i in sub.el_indices}
        filtered = GmshIO()
        filtered.nodes = gio.nodes
        filtered.physical = gio.physical
        filtered.elements = {eid: el for eid, el in gio.elements.items() if eid in keep_tags}

        if n_sub == 1:
            out_path = os.path.join(out_dir, "elements.msh")
        else:
            ix, iy, iz = np.unravel_index(i_sub, tuple(grid.shape))
            out_path = os.path.join(out_dir, f"subdomain_{ix}_{iy}_{iz}", "elements.msh")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        filtered.write(out_path)


def average_windows(output_mesh: Mesh, aperture_by_region_id: Dict[int, float],
                    grid: Grid,
                    young_rock: float, poisson_rock: float,
                    young_fracture: float, poisson_fracture: float,
                    bulk_region_id: int, fracture_region_ids: List[int] = None,
                    level: int = 2,
                    ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Volume-average strain and stress over the grid windows — one (eps_avg, sigma_avg) 3x3 pair
    per window, ordered as the grid cells (see module docstring for the <x> = sum/V formula).

    aperture_by_region_id is the UNDEFORMED per-fracture aperture (report sec. 3.1), computed
    ONCE (run_kubc) from the fracture radii — never `cross_section_updated`, which would
    reflect mechanical deformation. Keyed by REGION id, not element id: Flow123d renumbers
    elements in its own output (it does not preserve the pre-solve mesh's element ids — e.g.
    boundary-only faces are dropped from its own element count), so an element-id lookup built
    from a separately-loaded file would silently misalign; region ids are explicit DATA rather
    than an internal numbering scheme and ARE preserved, which is all this needs since aperture
    never varies within one fracture region anyway.
    """
    fracture_region_ids = fracture_region_ids or []

    # endorse readers; time 0.0 = first output time = the steady mechanics solution
    rid = np.asarray(output_mesh.get_p0_values("region_id", 0.0)).astype(int).ravel()
    stress9 = np.asarray(output_mesh.get_p0_values("stress", 0.0), dtype=float)
    assert stress9.shape[1] == 9, f"expected 9-component stress, got {stress9.shape}"

    # full-mesh strain field with the per-region compliance
    eps9 = np.zeros_like(stress9)
    bulk_rows = rid == bulk_region_id
    assert bulk_rows.any(), f"No bulk cells with region id {bulk_region_id} in the output."
    eps9[bulk_rows] = hooke_strain(
        stress9[bulk_rows].reshape(-1, 3, 3), young_rock, poisson_rock).reshape(-1, 9)
    if fracture_region_ids:
        frac_rows = np.isin(rid, fracture_region_ids)
        eps9[frac_rows] = hooke_strain(
            stress9[frac_rows].reshape(-1, 3, 3), young_fracture, poisson_fracture).reshape(-1, 9)

    macro_mesh = WindowsMacroMesh(grid)  # window bounds incl. half-open interior-interface rims
    V_ref = float(np.prod(grid.step))
    results = []
    for i_window in range(len(macro_mesh.elements)):
        sub, measures = _window_subdomain(output_mesh, rid, bulk_region_id, fracture_region_ids,
                                          aperture_by_region_id, macro_mesh, i_window, level)
        sigma_avg = sub.weighted_sum(stress9, measures=measures) / V_ref
        eps_avg = sub.weighted_sum(eps9, measures=measures) / V_ref
        results.append((eps_avg.reshape(3, 3), sigma_avg.reshape(3, 3)))
    return results
