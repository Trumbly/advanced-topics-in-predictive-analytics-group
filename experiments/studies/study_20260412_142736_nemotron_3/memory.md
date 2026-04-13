### Top-K Successful Experiments
- **exp_007** [completed]
  - arch: [resnet18] ResNet-18 via TorchvisionAdapter, baseline head | pretrained: resnet18
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: cmap_at_5=0.3195, f1_macro=0.2256, loss=0.2853, roc_auc_macro=0.9508
  - duration: 408.9s

- **exp_010** [completed]
  - arch: [mobilenet_v3_small] Lightweight MobileNet-V3 Small backbone with bottleneck attention | pretrained: mobilenet_v3_small
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: cmap_at_5=0.0124, f1_macro=0.0036, loss=0.3270, roc_auc_macro=0.6929
  - duration: 154.0s

- **exp_006** [completed]
  - arch: [cnn_attention] 4-conv CNN + self-attention → Linear
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: cmap_at_5=0.0198, f1_macro=0.0255, loss=0.5050, roc_auc_macro=0.6837
  - duration: 3271.9s

- **exp_009** [completed]
  - arch: [cnn_gru_hybrid] 3-conv CNN front-end feeding a 1-layer GRU(128) → Linear
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0, dropout=0.1
  - metrics: cmap_at_5=0.0125, f1_macro=0.0083, loss=0.5169, roc_auc_macro=0.6677
  - duration: 973.9s

- **exp_001** [completed]
  - arch: [custom_cnn] cnn_small_v1 baseline
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: cmap_at_5=0.0147, f1_macro=0.0123, loss=0.5325, roc_auc_macro=0.6538
  - duration: 0.0s

### Recent Failures
- **exp_008** [failed]
  - arch: [efficientnet_b0] EfficientNet-B0 via TorchvisionAdapter, baseline head | pretrained: efficientnet_b0
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1

- **exp_005** [failed]
  - arch: [mobilenet_v3_small] MobileNet-V3 Small via TorchvisionAdapter (lightweight backbone) | pretrained: mobilenet_v3_small
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1

- **exp_004** [failed]
  - arch: [cnn_gru_hybrid] 3-conv CNN front-end feeding 1-layer GRU(128) | pretrained: null
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1

- **exp_003** [failed]
  - arch: [cnn_attention] 2-conv CNN front-end with self-attention over channels
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1

- **exp_002** [failed]
  - arch: [mobilenet_v3_small] lightweight Conv2d backbone (1M params) with LazyLinear head | pretrained: mobilenet_v3_small
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1
