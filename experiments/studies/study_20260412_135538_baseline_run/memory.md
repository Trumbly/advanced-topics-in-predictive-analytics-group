### Top-K Successful Experiments
- **exp_001** [completed]
  - arch: [custom_cnn] cnn_small_v1 baseline
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: cmap_at_5=0.2183, f1_macro=0.1235, loss=1.1567, roc_auc_macro=0.6999
  - duration: 13.8s

### Recent Failures
- **exp_005** [failed]
  - arch: [cnn_attention] Deep residual CNN front-end (5 blocks) feeding into a Multi-Head Self-Attention layer, followed by Adaptive Pooling
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.2

- **exp_004** [failed]
