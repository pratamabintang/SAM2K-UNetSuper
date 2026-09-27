import os

# Penting di Windows agar .pyd menemukan DLL PyTorch/CUDA
os.add_dll_directory(r"E:\anaconda3\envs\landslide\bin")
os.add_dll_directory(r"E:\anaconda3\envs\landslide\Lib\site-packages\torch\lib")

import torch
from vmamba import SS2D


def main():
    print("=" * 70)
    print("SS2D / selective_scan_cuda_oflex SMOKE TEST")
    print("=" * 70)

    print("PyTorch :", torch.__version__)
    print("CUDA    :", torch.version.cuda)
    print("CUDA OK :", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia.")

    print("GPU     :", torch.cuda.get_device_name(0))

    # ------------------------------------------------------------
    # Input: [B, H, W, C] karena channel_first=False
    # ------------------------------------------------------------
    B, H, W, C = 2, 16, 16, 24

    x = torch.randn(
        B, H, W, C,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    print("\nInput:")
    print("  shape :", tuple(x.shape))
    print("  dtype :", x.dtype)
    print("  device:", x.device)

    # ------------------------------------------------------------
    # forward_type='v3' -> selective_scan_backend='oflex'
    # ------------------------------------------------------------
    model = SS2D(
        d_model=C,
        d_state=1,
        ssm_ratio=2.0,
        dt_rank="auto",
        d_conv=3,
        dropout=0.0,
        initialize="v0",
        forward_type="v3",
        channel_first=False,
    ).cuda()

    model.train()

    print("\nModel:")
    print("  class        :", type(model).__name__)
    print("  d_model      :", model.d_model)
    print("  d_state      :", model.d_state)
    print("  d_inner      :", model.d_inner)
    print("  forward_core :", model.forward_core)

    # ------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------
    print("\n[1] Forward ...")

    y = model(x)

    torch.cuda.synchronize()

    print("  output shape :", tuple(y.shape))
    print("  output dtype :", y.dtype)
    print("  output device:", y.device)
    print("  output finite:", torch.isfinite(y).all().item())
    print("  output mean  :", y.detach().mean().item())
    print("  output absmax:", y.detach().abs().max().item())

    assert y.shape == x.shape, (
        f"Shape mismatch: input={x.shape}, output={y.shape}"
    )

    assert torch.isfinite(y).all(), "Output mengandung NaN/Inf."

    # ------------------------------------------------------------
    # BACKWARD
    # ------------------------------------------------------------
    print("\n[2] Backward ...")

    loss = y.square().mean()

    print("  loss:", loss.item())

    loss.backward()

    torch.cuda.synchronize()

    print("  x.grad exists:", x.grad is not None)
    print("  x.grad finite:",
          torch.isfinite(x.grad).all().item())
    print("  x.grad absmax:",
          x.grad.abs().max().item())

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    # ------------------------------------------------------------
    # Check parameter gradients
    # ------------------------------------------------------------
    print("\n[3] Parameter gradients ...")

    bad = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"  {name:30s} grad=None")
            else:
                finite = torch.isfinite(param.grad).all().item()
                gmax = param.grad.detach().abs().max().item()

                print(
                    f"  {name:30s} "
                    f"shape={tuple(param.shape)!s:20s} "
                    f"grad_absmax={gmax:.6g} "
                    f"finite={finite}"
                )

                if not finite:
                    bad.append(name)

    if bad:
        raise RuntimeError(
            "Gradient NaN/Inf pada parameter: " + ", ".join(bad)
        )

    print("\n" + "=" * 70)
    print("SS2D SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()