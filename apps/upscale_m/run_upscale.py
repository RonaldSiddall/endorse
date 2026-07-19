"""
Main driver: buffered micro mesh -> six KUBC Flow123d runs -> inner-cube averaging -> effective
elastic tensor report.

Run (Docker Desktop must be running for the Flow123d wrapper):

    cd apps/upscale_m
    ../../venv/Scripts/python.exe run_upscale.py RUN_NAME [CONFIG]

Layout convention (source vs data, like Python_scripts' src/Raw_data split):
    configs/  - study definitions (CONFIG is looked up there; bare names work: config.yaml)
    runs/     - ALL run products, git-ignored; RUN_NAME becomes runs/RUN_NAME/ automatically
Each run dir contains the healed mesh, one directory per load case with all Flow123d
in/outputs (nested under kubc/ or subc/ per BC type — see kubc.py/subc.py), and the tensor
report(s) (name from config keys output.report_name_kubc/report_name_subc; all matrices are
the MEASURED inner-cube averages; prescribed load values are only boundary conditions, see
upscale_tensor.py).
"""
import logging
import os
import sys

_app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_app_dir, "src"))

from endorse import common

from micro_mesh import make_micro_mesh, make_averaging_grid
from kubc import run_kubc
from subc import run_subc
from postprocess import write_subdomain_meshes
from upscale_tensor import write_all_reports


def main(work_dir="default", config_file="config.yaml"):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    common.CallCache.instance()
    # configs live in configs/; accept both bare names and explicit paths
    cfg_path = os.path.join(_app_dir, "configs", config_file)
    if not os.path.isfile(cfg_path):
        cfg_path = os.path.join(_app_dir, config_file)
    cfg = common.load_config(cfg_path)
    # all run products go under runs/ (git-ignored), unless an absolute path is given
    if not os.path.isabs(work_dir):
        work_dir = os.path.join(_app_dir, "runs", work_dir)
    print(f"[upscale_m] config: {cfg_path}")
    print(f"[upscale_m] run dir: {work_dir}")

    bc_type = str(cfg.loads.bc_type).strip().lower()
    assert bc_type in ("kubc", "subc", "both"), \
        f"loads.bc_type must be 'kubc', 'subc' or 'both' (got {bc_type!r})"
    run_kubc_flag = bc_type in ("kubc", "both")
    run_subc_flag = bc_type in ("subc", "both")

    os.makedirs(work_dir, exist_ok=True)
    with common.workdir(work_dir, inputs=[]):
        print("=== upscale_m: micro mesh ===", flush=True)
        # subc_support builds the extra rigid-body support tetrahedra (micro_mesh.make_geometry)
        # into the SAME mesh whenever SUBC is needed, so a 'both' run solves both BC types on
        # identical geometry/averaging windows; left off entirely for a pure KUBC run.
        micro = make_micro_mesh(cfg.geometry, cfg.mesh, cfg.fractures,
                                cfg.materials.aperture_per_r, work_dir=".",
                                subc_support=run_subc_flag)
        grid = make_averaging_grid(cfg.geometry, micro)
        level = int(cfg.geometry.get("window_refine_level", 2))
        write_meshes = bool(cfg.output.get("write_subdomain_meshes", False))

        if run_kubc_flag:
            print("=== upscale_m: KUBC load cases ===", flush=True)
            kubc_results = run_kubc(cfg, micro, grid)
            print("=== upscale_m: KUBC tensor assembly ===", flush=True)
            write_all_reports(cfg, micro, grid, kubc_results, bc_type="kubc")
            if write_meshes:
                write_subdomain_meshes(micro, grid, level,
                                       float(cfg.materials.aperture_per_r), "kubc")

        if run_subc_flag:
            print("=== upscale_m: SUBC load cases ===", flush=True)
            subc_results = run_subc(cfg, micro, grid)
            print("=== upscale_m: SUBC tensor assembly ===", flush=True)
            write_all_reports(cfg, micro, grid, subc_results, bc_type="subc")
            if write_meshes:
                write_subdomain_meshes(micro, grid, level,
                                       float(cfg.materials.aperture_per_r), "subc")

    print("\n=== upscale_m: DONE ===")


if __name__ == "__main__":
    main(*sys.argv[1:])
