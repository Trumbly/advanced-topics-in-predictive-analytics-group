### Top-K Successful Experiments
- **exp_001** [completed]
  - arch: [custom_cnn] cnn_small_v1 baseline
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: loss=0.5097, roc_auc_macro=0.7144
  - duration: 158.5s

### Recent Failures
- **exp_006** [failed]
  - arch: [mobilenet_v3_small] MobileNetV3 backbone followed by a custom Squeeze-and-Excitation (SE) feature refinement block and Global Average Pooling | pretrained: mobilenet_v3_small
  - hyperparams: lr=0.0005, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0001, dropout=0.2

- **exp_005** [failed]
  - arch: [cnn_gru_hybrid] 3-conv CNN front-end (2-3 blocks) extracting spectral features, followed by a 1-layer GRU(256) processing the temporal sequence, and a final linear projection.
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0005, dropout=0.3

- **exp_004** [failed]
  - arch: [cnn_attention] EfficientNet-B0 backbone feeding into Self-Attention (Time-Domain) Encoder block | pretrained: efficientnet_b0
  - hyperparams: lr=0.0008, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0001, dropout=0.25
