# Mixed-Precision Stabilization and Selective Scan Precision Policy

## Context
During training of SAM2K-UNetSuper with SSF on Blackwell GPUs (SM120) under AMP FP16, intermittent NaN losses occurred at step 227 (end of Epoch 1) and intermittently in Epoch 2, accompanied by `Validation Loss: NaN` with a finite `IoU: 0.4783`. Investigation revealed that `DBISSF_Attention` invoked `selective_scan_fn_v1` without FP32 input promotion (unlike `CM_Attention`), causing recurrent hidden state accumulation across sequence length $L=16,384$ to exceed the FP16 ceiling ($65,504.0$). Additionally, spatial reductions in `structure_loss` over $512 \times 512 = 262,144$ elements saturated FP16 dynamic range.

## Decision
1. **Unify Selective Scan Autocast Protection**: Enforce that all invocations of `selective_scan_cuda` within SSF (`DBISSF_Attention` and `CM_Attention`) pass inputs in FP32 or via `SelectiveScan.apply` decorated with `@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)`.
2. **Switch AMP to BFloat16 on Modern Accelerators**: For SM80+ architectures (and specifically Blackwell SM120), adopt `torch.bfloat16` for autocast execution. BF16 shares the 8-bit exponent of FP32 ($10^{38}$ dynamic range), eliminating half-precision saturation while maintaining tensor core throughput without requiring dynamic loss scaling.
3. **Explicit FP32 Reductions in Loss Functions**: Force prediction and mask tensors to `torch.float32` prior to spatial pooling and weighting in `structure_loss`.
4. **Enforce Blacklist Filtering**: Ensure `blacklist_path` is never overridden to `None` in training scripts, keeping curated sensor noise out of training and validation.
