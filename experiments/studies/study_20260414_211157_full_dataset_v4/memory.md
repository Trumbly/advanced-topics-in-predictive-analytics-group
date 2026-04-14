### Top-K Successful Experiments
- **exp_001** [completed]
  - arch: [efficientnet_b0] pretrained EfficientNet-B0 baseline | pretrained: efficientnet_b0
  - hyperparams: lr=0.001, batch_size=128, epochs=4, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: cmap_at_5=0.5929, f1_macro=0.5858, loss=0.1414, roc_auc_macro=0.9825
  - duration: 5413.9s

- **exp_002** [completed]
  - arch: [mobilenet_v3_small] MobileNet-V3 Small via TorchvisionAdapter, baseline head | pretrained: mobilenet_v3_small
  - hyperparams: lr=0.001, batch_size=128, epochs=4, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: cmap_at_5=0.5508, f1_macro=0.5568, loss=0.1570, roc_auc_macro=0.9796
  - duration: 1209.9s

### Recent Failures
- **exp_003** [failed]
  - arch: [cnn_attention] 4-Conv block front-end (32->64->128) -> Temporal Self-Attention -> Global Average Pooling
  - hyperparams: lr=0.001, batch_size=128, epochs=4, optimizer=adam, weight_decay=0.0, dropout=0.1
