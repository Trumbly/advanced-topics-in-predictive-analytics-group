### Top-K Successful Experiments
- **exp_013** [completed]
  - arch: [efficientnet_b0] EfficientNet-B0 via TorchvisionAdapter, full backbone training from scratch
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.2
  - metrics: cmap_at_5=0.2453, f1_macro=0.0881, loss=1.1573, roc_auc_macro=0.6789
  - duration: 7.4s

- **exp_001** [completed]
  - arch: [custom_cnn] cnn_small_v1 baseline
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: cmap_at_5=0.1699, f1_macro=0.0595, loss=1.2104, roc_auc_macro=0.5877
  - duration: 114.2s

- **exp_012** [completed]
  - arch: [mobilenet_v3_small] MobileNetV3 Small via TorchvisionAdapter, optimized for spectrogram feature extraction | pretrained: mobilenet_v3_small
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.25
  - metrics: cmap_at_5=0.1008, f1_macro=0.0882, loss=1.1355, roc_auc_macro=0.4597
  - duration: 13.9s

- **exp_002** [completed]
  - arch: [cnn_attention] 3-conv CNN front-end followed by Self-Attention pooling layer
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: cmap_at_5=0.1432, f1_macro=0.1410, loss=1.1598, roc_auc_macro=0.4438
  - duration: 43.5s

- **exp_003** [completed]
  - arch: [cnn_gru_hybrid] 3-conv CNN front-end (32->64 channels) -> Global AvgPool over feature channels -> 2-layer GRU(128) head
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.2
  - metrics: loss=1.1361, roc_auc_macro=0.0000
  - duration: 0.0s

### Recent Failures
- **exp_016** [failed]
  - arch: [cnn_attention] MobileNetV3 Small backbone followed by a dedicated Time-Aware Self-Attention block | pretrained: mobilenet_v3_small
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.2

- **exp_015** [failed]
  - arch: [deep_cnn] 5-block deep residual-style CNN stack (32->64->128 channels) with BatchNorm and a final GlobalAveragePooling head
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.2

- **exp_014** [failed]
  - arch: [cnn_gru_hybrid] cnn_small_v1 backbone output features sequence fed into 2-layer GRU(128) head | pretrained: cnn_small_v1
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.3

- **exp_011** [failed]
  - arch: [specaugment_cnn] Custom 4-block CNN front-end with residual connections, followed by Global Average Pooling head
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.25

- **exp_010** [failed]
  - arch: [deep_cnn] 5-block residual CNN stack (32->64->128 channels) with BatchNorm and Global AvgPool head
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.2

- **exp_009** [failed]
  - arch: [mobilenet_v3_small] MobileNetV3 Small backbone adapted for spectrograms, followed by a dedicated Temporal Attention Pooling layer | pretrained: mobilenet_v3_small
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1

- **exp_008** [failed]
  - arch: [efficientnet_b0] EfficientNet-B0 via TorchvisionAdapter, standard feature extraction | pretrained: efficientnet_b0
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.15

- **exp_007** [failed]
  - arch: [deep_cnn] Custom 5-block residual CNN stack (32->64->128->256 channels)
  - hyperparams: lr=0.0008, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.25
