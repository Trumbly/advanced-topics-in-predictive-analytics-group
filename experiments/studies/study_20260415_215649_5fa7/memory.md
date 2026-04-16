### Top 4 by f1_macro
- `[resnet18] resnet18_specaugment_dp_safe_v2` [resnet18] — 0.9612 (exp_1d39957b8e)
- `[resnet18] resnet18_specaugment_dp_safe` [resnet18] — 0.3542 (exp_1143c179dd)
- `[resnet18] resnet18_specaugment_dp_safe` [resnet18] — 0.3417 (exp_203f2803d7)
- `[resnet18] resnet18_adapter_specaugment_dp_safe` [resnet18] — 0.0541 (exp_d7b74b03a1)

### Recent failures
- `[mobilenet_v3_small] mobilenet_v3_small_specaugment_dp_safe` — UnknownError: AttributeError: 'MobileNetV3' object has no attribute 'last_channel'
- `[resnet18] resnet18_specaugment_dp_safe_v3` — OOM: /bin/bash: line 1:     7 Killed                  "$ENTRYPOINT" "$A0" "$A1"