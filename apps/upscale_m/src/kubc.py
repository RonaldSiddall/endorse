"""
KUBC (kinematic uniform boundary conditions) load cases and the micro-problem runner.

STRUCTURE mirrors the endorse conductivity pipeline (src/endorse/macro_flow_model.py):
    run_kubc            ≈ macro_conductivity's load loop (gen_load_responses / micro_load_response)
    micro_problem       ≈ macro_flow_model.micro_problem — per-case workdir + call_flow +
                          micro_postprocess; endorse's `subproblem_input` step is intentionally
                          ABSENT: bulk and fracture materials are given by mesh REGIONS
                          substituted into the template, no per-element field file is needed.
    micro_postprocess   ≈ macro_flow_model.micro_postprocess — load the solver output mesh
                          (endorse mesh_class) and average the measured (load, response) pair,
                          here (eps, sigma), over the averaging windows (postprocess.average_windows
                          — a local port of R. Siddall's GeneralComputationClass, not endorse
                          Subproblems/Subdomain: those only support bulk (3D) elements, see
                          postprocess.py's module docstring and PLAN.md, 2026-07-14).

Six independent load states q = 1..6 prescribe u = E_q x on the whole outer boundary
(report sec. 2.5.2 / 3.2.1). Voigt ordering (report convention): [11, 22, 33, 23, 13, 12],
engineering shear (gamma = 2 eps) on the strain shear rows.

Each case runs Flow123d via endorse.common.call_flow with a single parametrized template
(flow123d_templates/, filename from config key loads.input_template_kubc); the displacement formula
string is generated from E_q, so additional (e.g. mixed) load states for a least-squares
identification need no new template.
"""
import bisect
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyvista as pv

from bgem.upscale import tn_to_voigt, voigt_to_tn
from bgem.upscale.fem import Grid
from endorse.common import dotdict, workdir, call_flow, report, FlowOutput
from endorse.mesh_class import load_mesh

from micro_mesh import MicroMesh
from postprocess import ENG_SHEAR, average_windows

_src_dir = os.path.dirname(os.path.abspath(__file__))
_app_dir = os.path.dirname(_src_dir)


def _displacement_point_data(mesh, time: float = 0.0) -> np.ndarray:
    """
    Flow123d writes 'displacement' as GMSH ElementNodeData (per-element corner values) — endorse
    mesh_class.Mesh only reads NodeData/ElementData, so there is no ready-made accessor for it.
    Average each node's value across the elements touching it into ONE (n_nodes, 3) array (a
    continuous P1 field, so all contributions at a shared node should already agree up to solver
    precision — averaging is just a safety net), in the SAME node order write_fields_vtu's
    pyvista grid uses (Mesh.nodes), so it can be attached as point data for ParaView's Warp By
    Vector filter (R. Siddall, 2026-07-14).
    """
    field_dict = mesh.gmsh_io.element_node_data["displacement"]
    times = [v.time for v in field_dict.values()]
    item = field_dict[bisect.bisect_right(times, time) - 1]
    sums = np.zeros((len(mesh.nodes), 3))
    counts = np.zeros(len(mesh.nodes))
    for el_tag, raw in zip(item.tags, item.values):
        node_idx = mesh.elements[mesh.el_indices[el_tag]].node_indices
        sums[node_idx] += np.asarray(raw).reshape(len(node_idx), -1)
        counts[node_idx] += 1
    return sums / np.maximum(counts, 1)[:, None]


def _write_mechanics_vtu(mesh, path: str, cell_fields: Dict[str, np.ndarray],
                         point_fields: Dict[str, np.ndarray],
                         smooth_fields: List[str] = ()) -> None:
    """
    Local port of endorse Mesh.write_fields_vtu (mesh_class.py) extended with POINT data:
    write_fields_vtu only ever writes cell_data (asserts field length == n_elements), which
    ParaView renders as one flat color per element ("faceted") — correct for a genuinely
    discontinuous field, but not how a continuous physical field like stress is meant to look.
    Extending write_fields_vtu itself lives in endorse, out of scope here (R. Siddall, 2026-07-14:
    no endorse edits) — kept as a small local duplicate of its ~10 lines instead.

    `smooth_fields` names cell_fields entries that ALSO get a nodally-averaged point_data copy
    (pyvista's own cell_data_to_point_data — reused, not reimplemented), so ParaView's default
    smooth/Gouraud shading applies to them directly — matching R. Siddall's original pipeline,
    whose .vtu files carried 'stress' as point data (Python_scripts/.../surface.py reads it
    straight out of point_data, converting the OTHER way only when it needs cell values for its
    own surface-integral method). The raw cell_data copy is kept alongside, unchanged, so the
    true per-element (unsmoothed) value stays inspectable. Categorical fields like region_id are
    deliberately NOT listed here — averaging a region id across a material boundary is
    meaningless (would show fractional, non-existent "regions" right where it matters most).
    """
    cells = [
        np.concatenate(([len(el.node_indices)], np.asarray(el.node_indices, dtype=np.int64)))
        for el in mesh.elements
    ]
    grid = pv.UnstructuredGrid(np.concatenate(cells), mesh._pv_celltypes(), mesh.nodes)
    for name, field in cell_fields.items():
        grid.cell_data[name] = np.asarray(field)
    for name, field in point_fields.items():
        grid.point_data[name] = np.asarray(field)
    if smooth_fields:
        smoothed = grid.cell_data_to_point_data()
        for name in smooth_fields:
            grid.point_data[name] = smoothed.point_data[name]
    grid.save(Path(path))


@dataclass
class LoadCaseResult:
    name: str
    prescribed_voigt: np.ndarray     # prescribed load (Voigt): strain E (KUBC) or stress sigma
                                      # (SUBC), engineering shear already applied if strain-type
    eps_voigt: np.ndarray            # measured strain per window, shape (n_windows, 6)
    sigma_voigt: np.ndarray          # measured stress per window, shape (n_windows, 6)
    spatial_file: str                # Flow123d output fields file the averages were read from


def make_load_case_result(name: str, load_tensor: np.ndarray, per_box, spatial_file: str,
                          load_is_strain: bool = True) -> LoadCaseResult:
    """
    Build a LoadCaseResult from one load case's prescribed load (KUBC: strain E; SUBC: stress
    sigma) and average_windows' list of per-window (eps, sigma) 3x3 pairs -- shared by run_kubc,
    subc.run_subc (after a real Flow123d solve) and debug_postprocess.main (against cached
    outputs), single place for the Voigt conversion and engineering shear gamma = 2 eps (report
    eq. 2.66, bgem tn_to_voigt, order [11,22,33,23,13,12]).
    `load_is_strain` (default True, so existing KUBC call sites are unaffected) selects whether
    the PRESCRIBED load gets the engineering-shear factor: correct for a strain Voigt vector,
    wrong for a stress one (Voigt shear stress has no such factor). The MEASURED eps/sigma are
    unaffected either way -- strain is always measured from displacement gradients and always
    gets ENG_SHEAR, regardless of which BC type produced it.
    """
    scale = ENG_SHEAR if load_is_strain else 1.0
    return LoadCaseResult(
        name=name,
        prescribed_voigt=tn_to_voigt(load_tensor[None])[0] * scale,
        eps_voigt=tn_to_voigt(np.array([e for e, s in per_box])) * ENG_SHEAR,
        sigma_voigt=tn_to_voigt(np.array([s for e, s in per_box])),
        spatial_file=spatial_file,
    )


@report
def micro_postprocess(mats: dotdict, micro: MicroMesh,
                      grid: Grid, level: int,
                      aperture_by_region_id: Dict[int, float], micro_model: FlowOutput):
    """
    Mirror of endorse macro_flow_model.micro_postprocess for the mechanics pair: load the
    solver's mechanics output (endorse mesh_class; stress, region_id, displacement — the
    FLOW/hydro sub-equation output is never read: its only job is carrying the cross_section
    FieldFE declaration required by Flow123d's schema, see PLAN.md), write a ParaView-viewable
    .vtu preview next to the GMSH output (_write_mechanics_vtu — a visualization convenience
    only, the computation itself stays GMSH-only) including displacement as POINT data so
    ParaView's Warp By Vector shows the deformed cube directly, and return the per-window
    averaged (eps, sigma) 3x3 pairs (postprocess.average_windows). stress additionally gets a
    smoothed point-data copy (region_id stays cell-only — see _write_mechanics_vtu) so the field
    renders as a continuous gradient instead of one flat color per element.
    """
    print("loading mesh:", micro_model.mechanic.spatial_file)
    output_mesh = load_mesh(micro_model.mechanic.spatial_file)
    _write_mechanics_vtu(
        output_mesh, "output/mechanics_fields.vtu",
        cell_fields=dict(
            stress=output_mesh.get_p0_values("stress", 0.0),
            region_id=output_mesh.get_p0_values("region_id", 0.0),
        ),
        point_fields=dict(
            displacement=_displacement_point_data(output_mesh, 0.0),
        ),
        smooth_fields=["stress"],
    )
    return average_windows(
        output_mesh, aperture_by_region_id, grid,
        young_rock=float(mats.young_modulus_rock),
        poisson_rock=float(mats.poisson_ratio_rock),
        young_fracture=float(mats.young_modulus_fracture),
        poisson_fracture=float(mats.poisson_ratio_fracture),
        bulk_region_id=micro.bulk_region_id,
        fracture_region_ids=micro.fracture_region_ids,
        level=level,
    )


def micro_problem(cfg: dotdict, tag: str, load_matrix: np.ndarray,
                  micro: MicroMesh, grid: Grid, level: int,
                  aperture_by_region_id: Dict[int, float]):
    """
    Mirror of endorse macro_flow_model.micro_problem: run one load case in its own work dir
    (template substitution + call_flow) and post-process. No `subproblem_input` — materials
    are region-wise template parameters and the per-fracture aperture travels as a SEPARATE
    cross_section field file (endorse fields_file + FieldFE), distinct from the mesh_file.
    Returns (per_window averages, output fields file path).
    """
    mats = cfg.materials
    mesh_abs_path = os.path.abspath(micro.mesh_file.path)
    cross_section_abs_path = os.path.abspath(micro.cross_section_file.path)
    boundary_regions = list(cfg.geometry.boundary_regions)
    missing = [r for r in boundary_regions if r not in micro.regions]
    assert not missing, f"geometry.boundary_regions not found in the mesh: {missing}"
    template_path = os.path.join(_app_dir, "flow123d_templates", cfg.loads.input_template_kubc)
    case_name = os.path.basename(tag)  # tag is namespaced, e.g. "kubc/pure_normal_E_11"
    with workdir(tag, inputs=[]):
        # NOTE: mesh copied explicitly, not via workdir(inputs=...) — workdir.copy uses src.stem
        # and silently drops the file extension (recorded in PLAN.md questions).
        shutil.copy2(mesh_abs_path, os.path.basename(mesh_abs_path))
        shutil.copy2(cross_section_abs_path, os.path.basename(cross_section_abs_path))
        # local per-case template copy so the rendered input carries the case name
        # (pure_normal_E_11.yaml, ...), mirroring the original per-case yaml files
        local_template = f"{case_name}_tmpl.yaml"
        shutil.copy2(template_path, local_template)
        params = dict(
            description=f"upscale_m KUBC load {tag}",
            mesh_file=os.path.basename(mesh_abs_path),
            cross_section_file=os.path.basename(cross_section_abs_path),
            bulk_regions="box",
            fracture_regions=f"[{', '.join(micro.fracture_regions)}]",
            boundary_regions=f"[{', '.join(boundary_regions)}]",
            # endorse micro-template idiom: "<matrix> @ [x,y,z]" in the template
            bc_displacement_matrix=str(load_matrix.tolist()),
            rock_young_modulus=f"{float(mats.young_modulus_rock):g}",
            rock_poisson_ratio=f"{float(mats.poisson_ratio_rock):g}",
            fracture_young_modulus=f"{float(mats.young_modulus_fracture):g}",
            fracture_poisson_ratio=f"{float(mats.poisson_ratio_fracture):g}",
        )
        micro_output = call_flow(cfg.machine_config, Path(local_template), params)
        # local_template's only job was naming the rendered input (call_flow strips "_tmpl.yaml"
        # to derive it, endorse.common.flow_call._prepare_inputs) — once that's done its raw,
        # unrendered content has zero further value (the original stays in flow123d_templates/)
        os.remove(local_template)
        if not micro_output.check_conv_reasons():
            raise ValueError(f"Load case {tag}: Flow123d simulation failed.")
        per_box = micro_postprocess(mats, micro, grid, level,
                                    aperture_by_region_id, micro_output)
        return per_box, os.path.abspath(micro_output.mechanic.spatial_file.path)


def load_cases(cfg: dotdict) -> List[Tuple[str, str, np.ndarray]]:
    """
    The six elementary prescribed macro-strain matrices E_q (report eq. 2.62), paired with their
    case/directory names — Voigt unit vectors mapped to tensors by bgem voigt_to_tn (report
    order), scaled by loads.deformation_parameter_beta. Single source of truth for run_kubc and
    debug_postprocess.main, which both need exactly this list.
    """
    beta = float(cfg.loads.deformation_parameter_beta)
    case_names = list(cfg.loads.case_names)
    case_dir_names = list(cfg.loads.case_dir_names)
    assert len(case_names) == 6 and len(case_dir_names) == 6, \
        "loads.case_names/case_dir_names must list exactly the 6 KUBC cases in Voigt order"
    return list(zip(case_names, case_dir_names, voigt_to_tn(beta * np.eye(6))))


def run_kubc(cfg: dotdict, micro: MicroMesh, grid: Grid) -> List[LoadCaseResult]:
    """
    Run the six KUBC load states on the given micro mesh (in the CURRENT work dir; one
    subdirectory per case) — the load loop of endorse macro_conductivity (gen_load_responses),
    returning per-case (load, response) = (eps, sigma) window averages. `grid` (bgem Grid over
    the inner cube, NOT imprinted in the mesh; consumed directly by postprocess.average_windows,
    no endorse Subproblems detour — see kubc.py's module docstring) is the caller's, so it is
    built exactly once and shared with the tensor-assembly step afterwards.
    """
    # sub-element refinement for elements cut by a window boundary
    level = int(cfg.geometry.get("window_refine_level", 2))
    # UNDEFORMED per-fracture aperture (report sec. 3.1), CONSTANT per fracture region, keyed by
    # region id (NOT element id: Flow123d renumbers elements internally in its own output, but
    # region ids are explicit DATA and ARE preserved -- see MicroMesh.aperture_by_region_id).
    aperture_by_region_id = micro.aperture_by_region_id(float(cfg.materials.aperture_per_r))

    results = []
    for name, dir_name, E in load_cases(cfg):
        tag = f"kubc/{dir_name}"
        print(f"[upscale_m kubc] case {name}: u = {E.tolist()} @ [x, y, z]")
        per_box, spatial_file = micro_problem(cfg, tag, E, micro, grid, level,
                                              aperture_by_region_id)
        results.append(make_load_case_result(name, E, per_box, spatial_file))
        n_sub = len(per_box)
        sig0 = results[-1].sigma_voigt[0]
        suffix = f" (window 0 of {n_sub})" if n_sub > 1 else ""
        print(f"[upscale_m kubc] case {name}: <sigma>{suffix} = {sig0}")
    return results
