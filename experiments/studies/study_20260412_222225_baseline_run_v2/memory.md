### Top-K Successful Experiments
- **exp_001** [completed]
  - arch: [custom_cnn] cnn_small_v1 baseline
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: cmap_at_5=0.2184, f1_macro=0.1037, loss=1.1283, roc_auc_macro=0.6676
  - duration: 9.0s

- **exp_010** [completed]
  - arch: [cnn_attention] 3-block Conv stack (64 channels) -> Self-Attention over time dimension -> Global Average Pooling -> Linear Head
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.2
  - metrics: ap_score_macro=0.1749, loss=0.9013, roc_auc_macro=0.5699
  - duration: 0.0s

- **exp_005** [completed]
  - arch: [efficientnet_b0] EfficientNet-B0 backbone adapted for spectrograms, followed by a Global Average Pooling layer for robust feature summarization before the final linear head. | pretrained: efficientnet_b0
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.15
  - metrics: cmap_at_5=0.1619, f1_macro=0.0862, loss=8.4285, roc_auc_macro=0.4542
  - duration: 6.2s

### Recent Failures
- **exp_014** [failed]
  - arch: [mobilenet_v3_small] MobileNetV3 Small backbone via TorchvisionAdapter, followed by a Global Average Pooling layer and a linear head for multi-label classification. | pretrained: mobilenet_v3_small
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.2

- **exp_013** [failed]
  - arch: [cnn_gru_hybrid] Lightweight 3-conv stack (64 channels) -> Flatten -> 1-layer GRU(128) -> Global Average Pooling -> Linear Head
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.2

- **exp_012** [failed]

- **exp_011** [failed]
  - arch: [deep_cnn] 5-layer residual CNN stack using BatchNorm and residual connections, followed by Global Average Pooling and the final linear head.
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.2

- **exp_009** [failed]

- **exp_008** [failed]
  - arch: [cnn_gru_hybrid] 3-conv CNN front-end (Conv2d stack) -> Global Pooling + Time-Series Feature Flattening -> 2-layer GRU(128) -> Linear Head
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1

- **exp_007** [failed]
  - arch: [cnn_attention] 3-conv front-end (BatchNorm+ReLU) -> Self-Attention Block (Time-domain context) -> Adaptive Pooling -> Linear Head
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1
