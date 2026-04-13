### Top-K Successful Experiments
- **exp_001** [completed]
  - arch: [custom_cnn] cnn_small_v1 baseline
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: loss=0.5064, roc_auc_macro=0.7019
  - duration: 202.0s

### Recent Failures
- **exp_005** [failed]
  - arch: [efficientnet_b0] EfficientNet-B0 backbone followed by a multi-head self-attention pooling layer before the final classification head | pretrained: efficientnet_b0
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.2

- **exp_004** [failed]
