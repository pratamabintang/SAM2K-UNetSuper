import os
import sys

# ============================================================
# DLL PATH
# ============================================================
os.add_dll_directory(
    r"E:\anaconda3\envs\landslide\bin"
)
os.add_dll_directory(
    r"E:\anaconda3\envs\landslide\Lib\site-packages\torch\lib"
)

# ============================================================
# PYTHON PATH
# ============================================================
ROOT = r"F:\LIDAR\PAKAGUS\SAM2K-UNetSuper\VMamba"
sys.path.insert(0, ROOT)

# ============================================================
# IMPORT
# ============================================================
import torch

print("=" * 70)
print("VMAMBA SELECTIVE SCAN SMOKE TEST")
print("=" * 70)

print("PyTorch       :", torch.__version__)
print("CUDA          :", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU           :", torch.cuda.get_device_name(0))

print()

# ============================================================
# IMPORT VMAMBA
# ============================================================
print("[1] Importing vmamba.py ...")

try:
    import vmamba
    print("    OK")
except Exception as e:
    print("    FAILED")
    print(type(e).__name__, ":", e)
    raise

print()

# ============================================================
# CHECK SELECTIVE SCAN MODULE
# ============================================================
print("[2] Checking selective_scan_cuda_oflex ...")

try:
    import selective_scan_cuda_oflex

    print("    OK")
    print("    module:", selective_scan_cuda_oflex)
    print("    fwd   :", selective_scan_cuda_oflex.fwd)
    print("    bwd   :", selective_scan_cuda_oflex.bwd)

except Exception as e:
    print("    FAILED")
    print(type(e).__name__, ":", e)
    raise

print()

# ============================================================
# FIND SS2D CLASS
# ============================================================
print("[3] Checking VMamba classes ...")

for name in [
    "SS2D",
    "SS2Dv0",
    "SS2Dv2",
    "SS2D_core",
]:
    if hasattr(vmamba, name):
        obj = getattr(vmamba, name)
        print(f"    {name}: FOUND -> {obj}")

print()

# ============================================================
# SHOW SELECTIVE SCAN CALL LOCATION
# ============================================================
print("[4] VMamba import completed.")
print()
print("Now we will inspect available classes.")
print()

# ============================================================
# PRINT CLASSES CONTAINING 'SS'
# ============================================================
print("[5] Classes containing 'SS':")

for name in dir(vmamba):
    if "SS" in name:
        obj = getattr(vmamba, name)
        if isinstance(obj, type):
            print("   ", name)

print()

print("=" * 70)
print("IMPORT SMOKE TEST PASSED")
print("=" * 70)