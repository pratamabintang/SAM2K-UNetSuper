import os
os.add_dll_directory(r"E:\anaconda3\envs\landslide\bin")
os.add_dll_directory(r"E:\anaconda3\envs\landslide\Lib\site-packages\torch\lib")

import torch
import selective_scan_cuda_core


def check(name, x):
    if x is None:
        print(f"{name}: None")
    else:
        print(
            f"{name}: shape={tuple(x.shape)}, "
            f"dtype={x.dtype}, "
            f"finite={torch.isfinite(x).all().item()}, "
            f"max_abs={x.abs().max().item():.6g}"
        )


print("=== ENV ===")
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

device = "cuda"

BATCH = 1
D = 4
L = 32
N = 1

print("\n=== INPUTS ===")

u = torch.randn(
    BATCH, D, L,
    device=device,
    dtype=torch.float32
) * 0.1

delta = torch.randn(
    BATCH, D, L,
    device=device,
    dtype=torch.float32
) * 0.05

A = -torch.rand(
    D, N,
    device=device,
    dtype=torch.float32
) - 0.1

B_param = torch.randn(
    BATCH, N, 1, L,
    device=device,
    dtype=torch.float32
) * 0.1

C_param = torch.randn(
    BATCH, N, 1, L,
    device=device,
    dtype=torch.float32
) * 0.1

D_param = torch.randn(
    D,
    device=device,
    dtype=torch.float32
) * 0.1

delta_bias = torch.zeros(
    D,
    device=device,
    dtype=torch.float32
)

for name, value in [
    ("u", u),
    ("delta", delta),
    ("A", A),
    ("B", B_param),
    ("C", C_param),
    ("D", D_param),
    ("delta_bias", delta_bias),
]:
    print(name, value.shape, value.dtype, value.device)


print("\n=== FWD ===")

delta_softplus = True

out, x = selective_scan_cuda_core.fwd(
    u,
    delta,
    A,
    B_param,
    C_param,
    D_param,
    delta_bias,
    delta_softplus,
    1,
)

check("out", out)
check("x", x)


print("\n=== DOUT ===")

# Gradient flowing from the next layer.
dout = torch.randn_like(out).contiguous()

check("dout", dout)


print("\n=== BWD ===")

grads = selective_scan_cuda_core.bwd(
    u,
    delta,
    A,
    B_param,
    C_param,
    D_param,
    delta_bias,
    dout,
    x,
    delta_softplus,
    1,
)

print("BWD returned:", type(grads))
print("number of outputs:", len(grads))

names = [
    "du",
    "ddelta",
    "dA",
    "dB",
    "dC",
    "dD",
    "ddelta_bias",
]

for name, value in zip(names, grads):
    check(name, value)

print("\n=== FINITE CHECK ===")

all_finite = True

for name, value in zip(names, grads):
    if value is not None:
        finite = torch.isfinite(value).all().item()
        print(f"{name}: {finite}")
        all_finite = all_finite and finite

print("\nALL GRADIENTS FINITE:", all_finite)

print("\n=== DONE ===")