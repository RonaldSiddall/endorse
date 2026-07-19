"""
SUBC (static uniform boundary conditions / traction) load cases and the micro-problem runner.

Mirrors kubc.py's shape and reuses its BC-agnostic pieces directly (micro_postprocess,
make_load_case_result, LoadCaseResult) rather than duplicating them — the only genuinely
SUBC-specific step is how a Voigt load turns into template parameters: KUBC substitutes ONE
displacement formula valid on the whole boundary (u = E x everywhere); SUBC computes a traction
t = sigma_q @ n_face PER FACE (6 separate values, since traction depends on the face normal,
unlike displacement) and needs the mesh built WITH the extra support tetrahedra
(micro_mesh.make_geometry(subc_support=True)) that remove the rigid-body modes a pure-traction
boundary otherwise leaves undetermined — the buffer's Saint-Venant decay does NOT substitute for
this, see PLAN.md.

Six independent load states q = 1..6 prescribe uniform traction on each of the 6 outer faces,
equivalent to a macroscopic stress state sigma_q (Voigt order [11,22,33,23,13,12], same
convention and case_names/case_dir_names as kubc.py — run_subc nests its case dirs under "subc/",
symmetric with run_kubc's own "kubc/" nesting, so a bc_type: both run keeps the two cleanly
separated).
"""
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from bgem.upscale import voigt_to_tn
from bgem.upscale.fem import Grid
from endorse.common import dotdict, workdir, call_flow

from kubc import LoadCaseResult, make_load_case_result, micro_postprocess
from micro_mesh import MicroMesh

_src_dir = os.path.dirname(os.path.abspath(__file__))
_app_dir = os.path.dirname(_src_dir)

# outward unit normal of each outer-cube face, box_with_sides/config.yaml boundary_regions
# naming convention (0 = minus/lower side, 1 = plus/upper side per axis)
_FACE_NORMALS = {
    "x0": np.array([-1.0, 0.0, 0.0]), "x1": np.array([1.0, 0.0, 0.0]),
    "y0": np.array([0.0, -1.0, 0.0]), "y1": np.array([0.0, 1.0, 0.0]),
    "z0": np.array([0.0, 0.0, -1.0]), "z1": np.array([0.0, 0.0, 1.0]),
}


def load_cases(cfg: dotdict) -> List[Tuple[str, str, np.ndarray]]:
    """
    The six elementary prescribed macro-stress tensors sigma_q, paired with their case/directory
    names — Voigt unit vectors mapped to tensors by bgem voigt_to_tn (the same generic
    Voigt<->tensor helper kubc.load_cases uses for strain — voigt_to_tn is just an index
    expansion, agnostic to what the 6 components physically mean), scaled by
    loads.stress_parameter_alpha. Case names/dirs are shared with KUBC (same six independent
    load states, same Voigt order) — only run_subc's directory nesting ("subc/") differs.
    """
    alpha = float(cfg.loads.stress_parameter_alpha)
    case_names = list(cfg.loads.case_names)
    case_dir_names = list(cfg.loads.case_dir_names)
    assert len(case_names) == 6 and len(case_dir_names) == 6, \
        "loads.case_names/case_dir_names must list exactly the 6 SUBC cases in Voigt order"
    return list(zip(case_names, case_dir_names, voigt_to_tn(alpha * np.eye(6))))


def micro_problem(cfg: dotdict, tag: str, load_matrix: np.ndarray,
                  micro: MicroMesh, grid: Grid, level: int,
                  aperture_by_region_id: Dict[int, float]):
    """
    Mirror of kubc.micro_problem: run one SUBC load case in its own work dir (template
    substitution + call_flow) and post-process (kubc.micro_postprocess, reused as-is — it only
    reads Flow123d's output fields, agnostic to which BC type produced them). The SUBC-specific
    part is computing each face's traction t = sigma_q @ n_face and widening bulk_regions to
    [box, support_tetras_volume] (SAME material as rock, per R. Siddall's original static_bc
    pipeline). Returns (per_window averages, output fields file path).
    """
    mats = cfg.materials
    mesh_abs_path = os.path.abspath(micro.mesh_file.path)
    cross_section_abs_path = os.path.abspath(micro.cross_section_file.path)
    template_path = os.path.join(_app_dir, "flow123d_templates", cfg.loads.input_template_subc)
    case_name = os.path.basename(tag)  # tag may be namespaced, e.g. "subc/pure_normal_E_11"
    with workdir(tag, inputs=[]):
        # NOTE: mesh copied explicitly, not via workdir(inputs=...) — see kubc.micro_problem's
        # identical note (workdir.copy uses src.stem and silently drops the file extension)
        shutil.copy2(mesh_abs_path, os.path.basename(mesh_abs_path))
        shutil.copy2(cross_section_abs_path, os.path.basename(cross_section_abs_path))
        local_template = f"{case_name}_tmpl.yaml"
        shutil.copy2(template_path, local_template)
        traction_params = {
            f"traction_{face}": str((load_matrix @ n).tolist())
            for face, n in _FACE_NORMALS.items()
        }
        params = dict(
            description=f"upscale_m SUBC load {tag}",
            mesh_file=os.path.basename(mesh_abs_path),
            cross_section_file=os.path.basename(cross_section_abs_path),
            bulk_regions="[box, support_tetras_volume]",
            fracture_regions=f"[{', '.join(micro.fracture_regions)}]",
            rock_young_modulus=f"{float(mats.young_modulus_rock):g}",
            rock_poisson_ratio=f"{float(mats.poisson_ratio_rock):g}",
            fracture_young_modulus=f"{float(mats.young_modulus_fracture):g}",
            fracture_poisson_ratio=f"{float(mats.poisson_ratio_fracture):g}",
            **traction_params,
        )
        micro_output = call_flow(cfg.machine_config, Path(local_template), params)
        os.remove(local_template)
        if not micro_output.check_conv_reasons():
            raise ValueError(f"Load case {tag}: Flow123d simulation failed.")
        per_box = micro_postprocess(mats, micro, grid, level,
                                    aperture_by_region_id, micro_output)
        return per_box, os.path.abspath(micro_output.mechanic.spatial_file.path)


def run_subc(cfg: dotdict, micro: MicroMesh, grid: Grid) -> List[LoadCaseResult]:
    """
    Run the six SUBC load states on the given micro mesh (must be built with
    micro_mesh.make_micro_mesh(..., subc_support=True) — see run_upscale.main), mirroring
    kubc.run_kubc. `micro`/`grid` are the caller's, shared with run_kubc for a bc_type: both run
    (same mesh, same averaging windows for both BC types).
    """
    level = int(cfg.geometry.get("window_refine_level", 2))
    aperture_by_region_id = micro.aperture_by_region_id(float(cfg.materials.aperture_per_r))

    results = []
    for name, dir_name, sigma in load_cases(cfg):
        tag = f"subc/{dir_name}"
        print(f"[upscale_m subc] case {name}: sigma = {sigma.tolist()}")
        per_box, spatial_file = micro_problem(cfg, tag, sigma, micro, grid, level,
                                              aperture_by_region_id)
        results.append(make_load_case_result(name, sigma, per_box, spatial_file,
                                             load_is_strain=False))
        n_sub = len(per_box)
        sig0 = results[-1].sigma_voigt[0]
        suffix = f" (window 0 of {n_sub})" if n_sub > 1 else ""
        print(f"[upscale_m subc] case {name}: <sigma>{suffix} = {sig0}")
    return results
