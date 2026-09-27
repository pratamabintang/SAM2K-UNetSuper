import os

# Windows DLL search path
os.add_dll_directory(r"E:\anaconda3\envs\landslide\bin")
os.add_dll_directory(r"E:\anaconda3\envs\landslide\Lib\site-packages\torch\lib")

import torch
from vmamba import VSSBlock


def main():
    print("=" * 70)
    print("VSSBlock / selective_scan_cuda_oflex SMOKE TEST")
    print("=" * 70)

    print("PyTorch :", torch.__version__)
    print("CUDA    :", torch.version.cuda)
    print("CUDA OK :", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia.")

    print("GPU     :", torch.cuda.get_device_name(0))

    # ------------------------------------------------------------
    # VSSBlock dengan channel_first=False
    #
    # VSSBlock menerima format:
    # [B, H, W, C]
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
    # VSSBlock
    #
    # forward_type='v3'
    # -> SS2D.forward_corev2()
    # -> selective_scan_backend='oflex'
    # ------------------------------------------------------------
    model = VSSBlock(
        hidden_dim=C,
        drop_path=0.0,
        channel_first=False,

        ssm_d_state=1,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v3",

        mlp_ratio=4.0,
        mlp_drop_rate=0.0,

        use_checkpoint=False,
        post_norm=False,
    ).cuda()

    model.train()

    print("\nModel:")
    print("  class       :", type(model).__name__)
    print("  hidden_dim  :", C)
    print("  ssm d_state :", model.op.d_state)
    print("  ssm d_inner :", model.op.d_inner)
    print("  forward     :", model.op.forward_core)

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
          x.grad.detach().abs().max().item())

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    # ------------------------------------------------------------
    # Parameter gradients
    # ------------------------------------------------------------
    print("\n[3] Parameter gradients ...")

    bad = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.grad is None:
            print(f"  {name:35s} grad=None")
            continue

        finite = torch.isfinite(param.grad).all().item()
        gmax = param.grad.detach().abs().max().item()

        print(
            f"  {name:35s} "
            f"shape={str(tuple(param.shape)):22s} "
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
    print("VSSBlock SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
