### Top-K Successful Experiments
- **exp_001** [completed]
  - arch: [efficientnet_b0] pretrained EfficientNet-B0 baseline | pretrained: efficientnet_b0
  - hyperparams: lr=0.001, batch_size=128, epochs=7, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: cmap_at_5=0.8058, f1_macro=0.5248, loss=0.0460, roc_auc_macro=0.9901
  - duration: 12960.9s

- **exp_002** [completed]
  - arch: [mobilenet_v3_small] MobileNetV3-Small via TorchvisionAdapter, baseline head | pretrained: mobilenet_v3_small
  - hyperparams: lr=0.0005, batch_size=128, epochs=7, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: cmap_at_5=0.6152, f1_macro=0.3232, loss=0.1231, roc_auc_macro=0.9831
  - duration: 2640.6s

- **exp_003** [completed]
  - arch: [cnn_attention] 3-conv CNN feature extractor feeding into a self-attention block, followed by Global Average Pooling
  - hyperparams: lr=0.001, batch_size=128, epochs=7, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: cmap_at_5=0.1860, f1_macro=0.1406, loss=0.3059, roc_auc_macro=0.9176
  - duration: 3358.4s

- **exp_005** [completed]
  - arch: [cnn_gru_hybrid] 3-block CNN feature extractor (32->64 channels) followed by nn.GRU(128) for temporal sequence modeling
  - hyperparams: lr=0.001, batch_size=128, epochs=7, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: cmap_at_5=0.0875, f1_macro=0.0764, loss=0.3689, roc_auc_macro=0.8706
  - duration: 10948.1s

### Recent Failures
- **exp_004** [failed]
  - arch: [deep_cnn] 4-block residual CNN backbone with BatchNorm and final attention pooling
  - hyperparams: lr=0.0008, batch_size=128, epochs=7, optimizer=adam, weight_decay=0.0, dropout=0.1
