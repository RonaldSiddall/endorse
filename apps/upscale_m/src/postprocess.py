"""
Volume integration of stress and strain over averaging windows.

Local port of R. Siddall's GeneralComputationClass (generate_mesh_data_in_one_file /
compute_partial_*_tensor), generalized with a window fractional-overlap weight so elements cut
by a window boundary enter with their captured volume/area fraction instead of an all-or-nothing
0/1 (R. Siddall's original always integrates the WHOLE RVE, one window, no cropping):

  <sigma> = (1/V) [ sum_bulk w_e sigma_e V_e  +  sum_frac w_f sigma_f A_f delta ]
  <eps>   = (1/V) [ sum_bulk w_e eps_e   V_e  +  sum_frac w_f eps_f   A_f delta ],   eps = M : sigma

R. Siddall, 14.7: endorse `homogenisation.Subdomain` cannot be reused here as-is for the fracture
(2D) side -- `Subdomain.create` hard-filters candidates to `mesh.el_dim_slice(dim=3)` and
`Mesh.el_volumes` returns 0.0 for anything but a 4-node tet (both in endorse/bgem, out of scope to
edit) -- and endorse's only other consumer of this machinery (macro_flow_model, conductivity) never
needs 2D elements either. `WindowSubdomain` below is ONE class/ONE weighted-sum formula for BOTH
element kinds (matching GeneralComputationClass's own pattern: read bulk and fracture SEPARATELY
because they need different measure formulas and material constants, then concatenate into one
array and integrate ONCE) — no separate bulk/fracture averaging code path.

Built on endorse/bgem primitives throughout: the window cut-fraction estimate reuses endorse
`macro_flow_model.refine_barycenters` (element split into equal-volume sub-simplices, weight =
fraction of sub-element barycenters inside the window; error ~ 2^-level), the candidate search is
endorse `Mesh.candidate_indices` (BIH), fields are read by endorse mesh_class P0 readers, Voigt
conversions by bgem tn_to_voigt.

The only local physics is `hooke_strain` — the isotropic inverse Hooke law
eps = ((1+nu) sigma - nu tr(sigma) I) / E (report sec. 3.4.2). AUDITED: no isotropic
stiffness/compliance builder exists anywhere in endorse or bgem (see PLAN.md log 2026-07-11).
Fracture elements are thin elastic plates weighted by the aperture delta (report sec. 3.1); the
adequacy of this treatment vs the explicit displacement-jump term J is an open question in PLAN.md.

Voigt convention (report): sigma = [11, 22, 33, 23, 13, 12]; strain shear rows carry engineering
shear gamma = 2 eps.
"""
import os
from functools import lru_cache
from typing import Dict, Iterator, List, Tuple

import attrs
import numpy as np

from bgem.gmsh.gmsh_io import GmshIO
from bgem.upscale.fem import Grid
from endorse.macro_flow_model import refine_barycenters
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


@lru_cache(maxsize=None)
def _simplex_barycentric_weights(n_vertices: int, level: int) -> np.ndarray:
    """
    Barycentric weights (n_sub, n_vertices) of the endorse refine_barycenters sub-element
    barycenters, computed ONCE on the reference simplex and mapped to any actual element by
    `weights @ element_vertices` (affine invariance of barycentric combinations).
    Needed because endorse refine_element asserts num_vertices == dim + 1 and thus rejects
    TRIANGLES EMBEDDED IN 3D, although it has the triangle refinement tables
    on the (k-1)-dimensional reference simplex the assert holds.
    """
    ref_simplex = np.eye(n_vertices)[:, 1:]  # rows: origin + unit vectors, shape (k, k-1)
    local = refine_barycenters(ref_simplex, level)
    return np.concatenate([1.0 - local.sum(axis=1, keepdims=True), local], axis=1)


def _window_weights(vertices: np.ndarray, lo_eff: np.ndarray, hi_eff: np.ndarray, level: int) -> np.ndarray:
    """
    Fraction of each simplex (shape (n_el, k, 3)) inside the half-open window (bounds carry the
    rim tolerance). Exact 1/0 shortcuts (window is convex); elements cut by the window boundary
    are estimated from the barycenters of the endorse refine_barycenters sub-elements. Generic in
    the vertex count k (3 for a fracture triangle, 4 for a bulk tet) via
    _simplex_barycentric_weights, so the SAME call estimates the window cut-fraction for either
    element kind — WindowSubdomain.create below calls it once per kind (numpy needs a uniform k
    per batch) and concatenates the results.
    """
    inside = lambda points: np.all((points > lo_eff) & (points < hi_eff), axis=-1)
    node_in = inside(vertices)
    weights = np.zeros(len(vertices))
    weights[np.all(node_in, axis=1)] = 1.0
    aabb_out = (np.any(np.min(vertices, axis=1) > hi_eff, axis=1)
                | np.any(np.max(vertices, axis=1) < lo_eff, axis=1))
    cut = np.where(~np.all(node_in, axis=1) & ~aabb_out)[0]
    if cut.size:
        w_ref = _simplex_barycentric_weights(vertices.shape[1], level)
        sub_barycenters = np.einsum("sk,nkd->nsd", w_ref, vertices[cut])
        weights[cut] = np.mean(inside(sub_barycenters), axis=1)
    return weights


def _tri_areas(vertices: np.ndarray) -> np.ndarray:
    """Area of each triangle, shape (n_el, 3, 3) -> (n_el,)."""
    cross = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


@attrs.define
class WindowSubdomain:
    """
    Elements captured by one averaging window — BULK (3D) and FRACTURE (2D) TOGETHER as a single
    per-element array (el_indices/intersect_weights/measures), local port of R. Siddall's
    GeneralComputationClass. ONE element "measure" regardless of dimension (true tet volume for
    bulk, area * aperture thin-plate overlay for fracture, report sec. 3.1) and ONE weighted_sum
    formula for both — bulk and fracture candidates are found/weighted separately inside create()
    only because they need different measure formulas (and numpy needs a uniform vertex count per
    batch), exactly as GeneralComputationClass reads bulk/fracture separately from the VTU (each
    needs its own compliance matrix) before concatenating into one array for the actual average.

    Window bounds are half-open ([lo, hi), outer rim inclusive) so a fracture lying exactly ON an
    interior window interface is counted in exactly one window; intersect_weights are the
    _window_weights volume/area-fraction estimate (elements cut by the window boundary enter with
    their captured fraction, not 0/1).
    """
    el_indices: np.ndarray
    intersect_weights: np.ndarray
    measures: np.ndarray

    @staticmethod
    def create(mesh: Mesh, region_id_field: np.ndarray, bulk_region_id: int,
              fracture_region_ids: List[int], aperture_by_region_id: Dict[int, float],
              lo_eff: np.ndarray, hi_eff: np.ndarray, level: int = 2) -> 'WindowSubdomain':
        relevant_ids = {bulk_region_id, *fracture_region_ids}
        candidates = np.array(mesh.candidate_indices(np.array([lo_eff, hi_eff])), dtype=int)
        candidates = candidates[np.isin(region_id_field[candidates], list(relevant_ids))]

        el_parts, w_parts, m_parts = [], [], []
        is_bulk = region_id_field[candidates] == bulk_region_id
        bulk_els, frac_els = candidates[is_bulk], candidates[~is_bulk]

        if bulk_els.size:
            vertices = np.array([mesh.elements[ie].vertices() for ie in bulk_els])
            w = _window_weights(vertices, lo_eff, hi_eff, level)
            keep = w > 0.0
            el_parts.append(bulk_els[keep])
            w_parts.append(w[keep])
            m_parts.append(mesh.el_volumes[bulk_els[keep]])

        if frac_els.size:
            vertices = np.array([mesh.elements[ie].vertices() for ie in frac_els])
            w = _window_weights(vertices, lo_eff, hi_eff, level)
            keep = w > 0.0
            kept_els = frac_els[keep]
            aperture = np.array([aperture_by_region_id[r] for r in region_id_field[kept_els]])
            el_parts.append(kept_els)
            w_parts.append(w[keep])
            m_parts.append(_tri_areas(vertices[keep]) * aperture)

        if not el_parts:
            return WindowSubdomain(np.array([], dtype=int), np.array([]), np.array([]))
        return WindowSubdomain(np.concatenate(el_parts), np.concatenate(w_parts),
                               np.concatenate(m_parts))

    def weighted_sum(self, element_vec_data: np.ndarray) -> np.ndarray:
        """
        Un-normalized volume/area-weighted sum — R. Siddall GeneralComputationClass's
        einsum('kij,k->ij', field, V_all) generalized with the window intersect_weights.
        """
        if self.el_indices.size == 0:
            return np.zeros(element_vec_data.shape[1])
        return (self.measures * self.intersect_weights) @ element_vec_data[self.el_indices]


def _iter_windows(grid: Grid) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """
    Per-window (i_sub, lo_eff, hi_eff) bounds — half-open [lo, hi), topmost grid faces inclusive
    (module docstring) — the ONE place this tolerance/rim computation lives, shared by
    average_windows and write_subdomain_meshes so they always select the SAME elements for a
    given window.
    """
    grid_hi = grid.origin + grid.dimensions
    half = grid.step / 2.0
    tol = 1e-8 * max(1.0, float(np.max(grid.step)))
    for i_sub, center in enumerate(grid.barycenters()):
        lo3, hi3 = center - half, center + half
        upper_rim = np.isclose(hi3, grid_hi, rtol=0.0, atol=tol)
        yield i_sub, lo3 - tol, np.where(upper_rim, hi3 + tol, hi3 - tol)


def write_subdomain_meshes(micro: MicroMesh, grid: Grid, level: int, aperture_per_r: float,
                           out_dir: str) -> None:
    """
    Diagnostic export (output.write_subdomain_meshes): one filtered .msh per averaging window,
    containing ONLY the elements WindowSubdomain.create selects for that window (same selection
    average_windows uses, via the shared _iter_windows bounds) — so a window's effective-tensor
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
    n_sub = int(grid.n_elements)

    for i_sub, lo_eff, hi_eff in _iter_windows(grid):
        window = WindowSubdomain.create(mesh, region_id_field, micro.bulk_region_id,
                                        micro.fracture_region_ids, aperture_by_region_id,
                                        lo_eff, hi_eff, level)
        keep_tags = {mesh.el_ids[i] for i in window.el_indices}
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
    per window, ordered as the grid cells:

      <x> = WindowSubdomain_sum(x) / V_ref,  V_ref = window volume (R. Siddall's
      GeneralComputationClass._V_RVE with use_actual_volume=False: divide by the DECLARED window
      volume, not the captured element volume — the two agree up to the O(2^-level) cut-boundary
      error of the window-weight estimate itself, since a fracture has zero true 3D volume and
      bulk tets exactly tile the window).

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

    # topmost faces of the whole grid stay INCLUSIVE, interior interfaces half-open [lo, hi)
    V_ref = float(np.prod(grid.step))
    results = []
    for i_window, lo_eff, hi_eff in _iter_windows(grid):
        window = WindowSubdomain.create(output_mesh, rid, bulk_region_id, fracture_region_ids,
                                        aperture_by_region_id, lo_eff, hi_eff, level)
        sigma_avg = window.weighted_sum(stress9) / V_ref
        eps_avg = window.weighted_sum(eps9) / V_ref
        results.append((eps_avg.reshape(3, 3), sigma_avg.reshape(3, 3)))
    return results
