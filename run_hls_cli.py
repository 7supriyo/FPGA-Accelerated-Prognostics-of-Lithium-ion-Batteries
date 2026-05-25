import os
import shutil
import stat
from pathlib import Path

import vitis

ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT / "cli_workspace"
WORKSPACE_NAME = os.environ.get("HLS_WORKSPACE_NAME", "w")
WORKSPACE = WORKSPACE_ROOT / WORKSPACE_NAME
CFG_FILE = ROOT / "hls_component.cfg"
COMPONENT_NAME = "battery_xgb_hls_comp"
PART = os.environ.get("HLS_PART", "xc7z020clg400-1")
CLOCK_NS = os.environ.get("HLS_CLOCK_PERIOD", "10.0")
FLOW = os.environ.get("HLS_FLOW", "all").strip().lower()


def force_remove_tree(path: Path) -> None:
    def handle_remove_readonly(func, target, exc_info):
        os.chmod(target, stat.S_IWRITE)
        func(target)

    shutil.rmtree(path, onexc=handle_remove_readonly)


if WORKSPACE.exists():
    force_remove_tree(WORKSPACE)

WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

client = vitis.create_client()
client.set_workspace(str(WORKSPACE))

print(f"Workspace : {WORKSPACE}")
print(f"Component : {COMPONENT_NAME}")
print(f"Config    : {CFG_FILE}")

component = client.create_hls_component(
    name=COMPONENT_NAME,
    part=PART,
    cfg_file=str(CFG_FILE),
)

if FLOW in ("csim", "all"):
    print("\n--- Running C simulation ---")
    component.run(operation="C_SIMULATION")

if FLOW in ("synth", "all"):
    print("\n--- Running synthesis ---")
    component.run(operation="SYNTHESIS")

if FLOW in ("cosim", "all"):
    print("\n--- Running C/RTL cosimulation ---")
    component.run(operation="CO_SIMULATION")

vitis.dispose()
print("\nVitis CLI HLS flow complete.")
