import os

os.add_dll_directory(r"E:\anaconda3\envs\landslide\bin")
os.add_dll_directory(
    r"E:\anaconda3\envs\landslide\Lib\site-packages\torch\lib"
)

import torch
import selective_scan_cuda_core


print("=== ENV ===")
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print()

print("=== MODULE ===")
print("fwd:", hasattr(selective_scan_cuda_core, "fwd"))
print("bwd:", hasattr(selective_scan_cuda_core, "bwd"))
print()

# Minimal configuration
batch = 1
dim = 4
seqlen = 32
dstate = 1

dtype = torch.float32
device = "cuda"

u = torch.randn(
    batch, dim, seqlen,
    device=device,
    dtype=dtype,
)

delta = torch.randn(
    batch, dim, seqlen,
    device=device,
    dtype=dtype,
)

A = torch.randn(
    dim, dstate,
    device=device,
    dtype=dtype,
)

B = torch.randn(
    batch, 1, dstate, seqlen,
    device=device,
    dtype=dtype,
)

C = torch.randn(
    batch, 1, dstate, seqlen,
    device=device,
    dtype=dtype,
)

D = torch.randn(
    dim,
    device=device,
    dtype=dtype,
)

delta_bias = torch.randn(
    dim,
    device=device,
    dtype=dtype,
)

u.requires_grad_(True)
delta.requires_grad_(True)
A.requires_grad_(True)
B.requires_grad_(True)
C.requires_grad_(True)
D.requires_grad_(True)
delta_bias.requires_grad_(True)

print("=== INPUTS ===")
for name, x in [
    ("u", u),
    ("delta", delta),
    ("A", A),
    ("B", B),
    ("C", C),
    ("D", D),
    ("delta_bias", delta_bias),
]:
    print(name, tuple(x.shape), x.dtype, x.device)

print()
print("=== FWD ===")

try:
    result = selective_scan_cuda_core.fwd(
        u,
        delta,
        A,
        B,
        C,
        D,
        delta_bias,
        True,
        1,
    )

    print("FWD returned:", type(result))
    print("number of outputs:", len(result))

    for i, x in enumerate(result):
        if torch.is_tensor(x):
            print(
                f"output[{i}]:",
                tuple(x.shape),
                x.dtype,
                "finite=",
                bool(torch.isfinite(x).all()),
            )
        else:
            print(f"output[{i}]:", type(x))

except Exception as e:
    print("FWD ERROR:")
    print(type(e).__name__, e)
    raise

print()
print("=== DONE ===")