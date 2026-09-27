import os

# ============================================================
# DLL PATH — supaya selective_scan_cuda_oflex bisa diload
# ============================================================
os.add_dll_directory(r"E:\anaconda3\envs\landslide\bin")
os.add_dll_directory(
    r"E:\anaconda3\envs\landslide\Lib\site-packages\torch\lib"
)

import torch

from vmamba import Backbone_VSSM


def check_finite(name, x):
    ok = torch.isfinite(x).all().item()
    print(
        f"  {name:<12} "
        f"shape={tuple(x.shape)!s:<24} "
        f"dtype={str(x.dtype):<16} "
        f"device={str(x.device):<8} "
        f"finite={ok} "
        f"absmax={x.detach().abs().max().item():.6g}"
    )
    assert ok, f"{name} contains NaN/Inf"


print("=" * 70)
print("Backbone_VSSM / selective_scan_cuda_oflex SMOKE TEST")
print("=" * 70)

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.version.cuda}")
print(f"CUDA OK : {torch.cuda.is_available()}")

assert torch.cuda.is_available(), "CUDA tidak tersedia"

device = torch.device("cuda")
print(f"GPU     : {torch.cuda.get_device_name(0)}")

# ============================================================
# Input
#
# Kita gunakan image kecil dulu supaya VRAM aman.
# ============================================================
B = 2
C = 3
H = 128
W = 128

x = torch.randn(
    B, C, H, W,
    device=device,
    dtype=torch.float32,
    requires_grad=True,
)

print("\nInput:")
print(f"  shape : {tuple(x.shape)}")
print(f"  dtype : {x.dtype}")
print(f"  device: {x.device}")

# ============================================================
# Backbone
#
# Dibuat lebih kecil daripada konfigurasi ImageNet default
# agar smoke test ringan.
#
# dims harus mengikuti jumlah stage = len(depths).
# ============================================================
model = Backbone_VSSM(
    patch_size=4,
    in_chans=3,

    depths=[1, 1, 1, 1],
    dims=[24, 48, 96, 192],

    ssm_d_state=1,
    ssm_ratio=2.0,
    ssm_dt_rank="auto",
    ssm_act_layer="silu",
    ssm_conv=3,

    forward_type="v3",

    mlp_ratio=4.0,
    drop_path_rate=0.0,

    patch_norm=True,
    norm_layer="ln",

    downsample_version="v2",
    patchembed_version="v1",

    use_checkpoint=False,

    out_indices=(0, 1, 2, 3),
)

model = model.to(device)
model.train()

print("\nModel:")
print(f"  class       : {model.__class__.__name__}")
print("  stages      :", len(model.layers))
print("  out_indices :", model.out_indices)
print("  channel_first:", model.channel_first)

for i, layer in enumerate(model.layers):
    print(
        f"  stage{i}: "
        f"blocks={len(layer.blocks)} "
        f"downsample={layer.downsample.__class__.__name__}"
    )
print(f"  out_indices : {model.out_indices}")
print(f"  channel_first: {model.channel_first}")

# ============================================================
# Forward
# ============================================================
print("\n[1] Forward ...")

outs = model(x)

assert isinstance(outs, (list, tuple)), \
    f"Backbone output harus list/tuple, dapat {type(outs)}"

print(f"  number of outputs: {len(outs)}")

assert len(outs) == 4, \
    f"Expected 4 feature maps, got {len(outs)}"

for i, out in enumerate(outs):
    check_finite(f"stage{i}", out)

# ============================================================
# Shape checks
# ============================================================
print("\n[2] Feature-map shape checks ...")

expected = [
    (B, 24, 32, 32),
    (B, 48, 16, 16),
    (B, 96, 8, 8),
    (B, 192, 4, 4),
]

for i, (out, exp) in enumerate(zip(outs, expected)):
    print(f"  stage{i}: actual={tuple(out.shape)}, expected={exp}")

    assert tuple(out.shape) == exp, (
        f"stage{i} shape mismatch: "
        f"got {tuple(out.shape)}, expected {exp}"
    )

print("  Shape checks PASSED")

# ============================================================
# Backward
#
# Gunakan semua stage agar seluruh backbone ikut mendapat
# gradient.
# ============================================================
print("\n[3] Backward ...")

loss = sum(out.float().mean() for out in outs)

print(f"  loss: {loss.item()}")

loss.backward()

assert x.grad is not None, "Input gradient tidak ada"
assert torch.isfinite(x.grad).all(), "Input gradient NaN/Inf"

print("  x.grad exists : True")
print("  x.grad finite : True")
print(f"  x.grad absmax : {x.grad.abs().max().item():.6g}")

# ============================================================
# Parameter gradients
# ============================================================
print("\n[4] Parameter gradients ...")

n_param = 0
n_grad = 0

for name, param in model.named_parameters():

    if not param.requires_grad:
        continue

    n_param += 1

    if param.grad is None:
        print(f"  MISSING GRAD  {name}")
        continue

    n_grad += 1

    finite = torch.isfinite(param.grad).all().item()

    print(
        f"  {name:<45} "
        f"shape={str(tuple(param.shape)):<20} "
        f"grad_absmax={param.grad.abs().max().item():.6g} "
        f"finite={finite}"
    )

    assert finite, f"NaN/Inf gradient: {name}"

print("\nGradient summary:")
print(f"  trainable parameters : {n_param}")
print(f"  parameters w/ grad   : {n_grad}")

assert n_grad == n_param, (
    f"Tidak semua parameter mendapat gradient: "
    f"{n_grad}/{n_param}"
)

# ============================================================
# CUDA memory
# ============================================================
print("\n[5] CUDA memory ...")

torch.cuda.synchronize()

allocated = torch.cuda.memory_allocated() / 1024**2
reserved = torch.cuda.memory_reserved() / 1024**2

print(f"  allocated : {allocated:.2f} MB")
print(f"  reserved  : {reserved:.2f} MB")

# ============================================================
# PASS
# ============================================================
print("\n" + "=" * 70)
print("BACKBONE_VSSM SMOKE TEST PASSED")
print("=" * 70)