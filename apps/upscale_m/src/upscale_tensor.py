"""
Assembly of the effective elastic tensor from the six load-case results.

Method: least-squares identification of the linear map responses = C @ loads via the endorse
equivalent-tensor machinery (`endorse.equivalent_field.eq_tensor`, dimension-generic; used here
with D = 6 on the Voigt vectors). With exactly 6 independent loads the LS solution IS the exact
6-case inversion C = Sigma @ E^-1 (report sec. 2.5.2); more (e.g. mixed) load states extend it
to a genuine LS fit with no code change. BOTH matrices are MEASURED: column q of E and Sigma
holds the volume-averaged strain/stress over the inner cube for load case q. The prescribed load
values are deliberately NOT used as results: with fractures intersecting the outer boundary the
average strain theorem fails (displacement jumps contribute on the boundary), so <eps> over the
averaging volume is the only consistent strain measure. The prescribed formula remains what it
physically is — just the boundary condition.

(The SYMMETRY-constrained fit — 21 unknowns — needs a dim-6 Voigt-pair table in endorse
eq_tensor.tn_homo_kernel_sym, which currently stops at dim 3: upstream question in PLAN.md.)

The report format mirrors R. Siddall's original GenerateKinematicEffectiveElasticTensor output
(sections Sigma, E, C_k, S_k; same prefixes and rulers).
"""
import os
from datetime import datetime
from typing import List, Tuple

import numpy as np

from endorse.equivalent_field import eq_tensor


def assemble_matrices(results: List["LoadCaseResult"], i_sub: int = 0):
    """
    Column-stack the measured per-case Voigt vectors of subdomain `i_sub`:
    returns (E, Sigma), each 6x6, column q = load case q.
    """
    E = np.column_stack([np.atleast_2d(r.eps_voigt)[i_sub] for r in results])
    Sigma = np.column_stack([np.atleast_2d(r.sigma_voigt)[i_sub] for r in results])
    return E, Sigma


def equivalent_tensor(loads: np.ndarray, responses: np.ndarray) -> np.ndarray:
    """
    Effective tensor C (6x6) from load/response COLUMN matrices (responses = C @ loads),
    via endorse eq_tensor LS (rows of eq_tensor input = load cases, hence the transposes).
    Exactly 6 independent loads -> unique solution = the exact inversion.
    """
    assert loads.shape[0] == 6 and loads.shape == responses.shape, \
        f"need 6-row Voigt column matrices, got {loads.shape} / {responses.shape}"
    eq = eq_tensor(dim=6)  # unconstrained 36-unknown kernel; 'sym' pending upstream (PLAN.md)
    C_flat = eq.flat(loads.T, responses.T)
    return eq.to_full_tn(C_flat).reshape(6, 6)


def spectral_properties(M: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Eigenvalues and spectral norm ||M||_2 of a general square matrix (R. Siddall's SVD note:
    ||M||_2 = sigma_max(M) = sqrt(lambda_max(M^T M)), which reduces to max|eigenvalue| only for
    a SYMMETRIC M). C_k is not exactly symmetric -- least-squares noise from the 6-case fit, small
    but nonzero (see PLAN.md's [1,1,1] vs [2,2,2] check) -- so eigenvalues are the general spectrum
    of C_k itself (np.linalg.eigvals; may come out as a complex-conjugate pair when that asymmetry
    is present) and the norm goes via SVD (np.linalg.norm(..., ord=2)), NOT via max|eigenvalue|.
    Eigenvalues are returned as a COLUMN vector, shape (n, 1) -- matching the LaTeX convention of
    stacking vectors as columns, e.g. V = [v_1 v_2 ... v_n].
    """
    eigenvalues = np.linalg.eigvals(M).reshape(-1, 1)
    spectral_norm = np.linalg.norm(M, ord=2)
    return eigenvalues, spectral_norm


def _write_vector(f, title, symbol, v, title_indent=26):
    """Column layout, one entry per row -- mirrors _write_matrix's bracket/prefix convention."""
    f.write("=" * 118 + "\n")
    f.write(" " * title_indent + title + "\n")
    f.write("=" * 118 + "\n\n")
    v = np.ravel(v)
    mid = len(v) // 2
    for i, x in enumerate(v):
        prefix = f"{symbol} =".ljust(8) if i == mid else " " * 8
        entry = f"{x.real:+.6e}{x.imag:+.6e}j" if np.iscomplexobj(v) else f"{x:.6e}"
        f.write(f"{prefix} [ {entry:>28s} ]\n")
    f.write("\n\n")


_BC_REPORT_INFO = {
    "kubc": ("kinematic boundary conditions", "k", "report_name_kubc"),
    "subc": ("static boundary conditions", "s", "report_name_subc"),
}


def write_all_reports(cfg, micro, grid, results: List["LoadCaseResult"],
                      bc_type: str = "kubc") -> List[np.ndarray]:
    """
    Assemble and write the effective-tensor report for every averaging window in `grid` -- single
    source of truth for the tensor-assembly step, used identically by run_upscale.py (after a real
    Flow123d run) and debug_postprocess.py (against cached outputs, no Docker): both already build
    the same `micro`/`grid` beforehand, so recomputing the reports for an existing run needs no
    resimulation, just re-running this loop against the cached mechanics_fields.msh files.
    `bc_type` ("kubc", default, or "subc") selects the label/symbol (C_k/S_k vs C_s/S_s) AND the
    report location: report(s) are written under a bc_type/ subdirectory (kubc/ or subc/),
    alongside that BC type's own case directories (kubc.run_kubc/subc.run_subc — same "kubc/" /
    "subc/" nesting), so a bc_type: both run keeps everything for each BC type together in one
    place instead of colliding or scattering the reports at the run root.
    Returns the list of C_<suffix> matrices, one per window (grid order).
    """
    bc_label, symbol_suffix, report_name_key = _BC_REPORT_INFO[bc_type]
    report_name = cfg.output[report_name_key]
    subdivision = [int(v) for v in cfg.geometry.get("subdomains", [1, 1, 1])]
    geo_radii = micro.fracture_radii
    meta = dict(
        L_inner=cfg.geometry.L_inner,
        L_ext_factor=cfg.geometry.get("L_ext_factor", 2.0),
        subdomains=subdivision,
        beta=cfg.loads.deformation_parameter_beta,
        alpha=cfg.loads.stress_parameter_alpha,
        E_rock=cfg.materials.young_modulus_rock,
        nu_rock=cfg.materials.poisson_ratio_rock,
        E_fracture=cfg.materials.young_modulus_fracture,
        nu_fracture=cfg.materials.poisson_ratio_fracture,
        fractures=[(float(rho), list(map(float, center)), list(map(float, normal)))
                   for rho, center, normal
                   in zip(geo_radii, micro.fractures.center, micro.fractures.normal)],
    )
    n_sub = int(grid.n_elements)
    all_C = []
    for i_sub, center in enumerate(grid.barycenters()):
        lo3, hi3 = center - grid.step / 2.0, center + grid.step / 2.0
        E, Sigma = assemble_matrices(results, i_sub)
        C_k = equivalent_tensor(E, Sigma)
        meta_i = dict(meta)
        meta_i["subdomain_box"] = [list(np.round(lo3, 6)), list(np.round(hi3, 6))]
        if n_sub == 1:
            report_path = os.path.join(bc_type, report_name)
        else:
            ix, iy, iz = np.unravel_index(i_sub, tuple(grid.shape))
            report_path = os.path.join(bc_type, f"subdomain_{ix}_{iy}_{iz}", report_name)
        write_report(report_path, results, E, Sigma, C_k, meta_i,
                    bc_label=bc_label, symbol_suffix=symbol_suffix)
        all_C.append(C_k)

        if n_sub == 1:
            with np.printoptions(precision=3, suppress=False, linewidth=140):
                print(f"\nC_{symbol_suffix}:\n", C_k)
        else:
            ix, iy, iz = np.unravel_index(i_sub, tuple(grid.shape))
            print(f"  subdomain ({ix},{iy},{iz}): "
                  f"C11 = {C_k[0, 0]:.4e}  C33 = {C_k[2, 2]:.4e}  C66 = {C_k[5, 5]:.4e}")
    return all_C


def _write_matrix(f, title, symbol, M, title_indent=26):
    f.write("=" * 118 + "\n")
    f.write(" " * title_indent + title + "\n")
    f.write("=" * 118 + "\n\n")
    for i in range(6):
        prefix = f"{symbol} =".ljust(8) if i == 2 else " " * 8
        row = " ".join([f"{M[i, j]:>16.6e}" for j in range(6)])
        f.write(f"{prefix} [ {row} ]\n")
    f.write("\n\n")


def write_report(path: str, results, E, Sigma, C, meta: dict,
                 bc_label: str = "kinematic boundary conditions", symbol_suffix: str = "k"):
    """
    Sigma, E (measured), C_<suffix>, S_<suffix>, eigenvalues/spectral norm of C_<suffix>, source
    VTU list.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    S = np.linalg.inv(C)
    eigenvalues, spectral_norm = spectral_properties(C)
    c_sym, s_sym = f"C_{symbol_suffix}", f"S_{symbol_suffix}"
    with open(path, "w") as f:
        f.write(f"upscale_m effective elastic tensor report ({bc_label})\n")
        f.write(f"generated: {datetime.now().isoformat(timespec='seconds')}\n")
        for key, value in meta.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        _write_matrix(f, f"Macroscopic stress matrix Sigma for {bc_label}", "Sigma", Sigma)
        _write_matrix(f, f"Macroscopic deformation matrix E for {bc_label}", "E", E)
        _write_matrix(f, f"Effective Elastic Tensor {c_sym} (with dash) for {bc_label}",
                      c_sym, C, title_indent=18)
        _write_matrix(f, f"Compliance Tensor {s_sym} (inverse of {c_sym}) for {bc_label}",
                      s_sym, S, title_indent=18)
        _write_vector(f, f"Eigenvalues of {c_sym} for {bc_label}", f"eig({c_sym})", eigenvalues,
                     title_indent=18)
        f.write(f"Spectral norm ||{c_sym}||_2 = {spectral_norm:.6e}\n\n")

        f.write("-" * 118 + "\n")
        f.write("Result computed using the following Flow123d output files:\n")
        for r in results:
            f.write(f"- [{r.name}] {r.spatial_file}\n")
    print(f"[upscale_m] tensor report written: {os.path.abspath(path)}")
