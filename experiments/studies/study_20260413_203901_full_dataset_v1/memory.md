### Top-K Successful Experiments
- **exp_002** [completed]
  - arch: [efficientnet_b0] EfficientNet-B0 via TorchvisionAdapter, baseline head | pretrained: efficientnet_b0
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: cmap_at_5=0.3697, f1_macro=0.1800, loss=0.2901, roc_auc_macro=0.9590
  - duration: 1851.4s

- **exp_001** [completed]
  - arch: [custom_cnn] cnn_small_v1 baseline
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: cmap_at_5=0.0601, f1_macro=0.0627, loss=0.4512, roc_auc_macro=0.8298
  - duration: 500.2s
