### Top-K Successful Experiments
- **exp_005** [completed]
  - arch: [custom_cnn] 4-conv from-scratch with BatchNorm, SE attention, and multi-scale pooling
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.3
  - metrics: cmap_at_5=0.6746, f1_macro=0.4353, loss=0.1244, roc_auc_macro=0.9806
  - duration: 7149.9s

- **exp_003** [completed]
  - arch: [deep_cnn] 5-conv deep CNN with residual connections, BatchNorm, and SE attention blocks, followed by adaptive pooling and sigmoid output
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.3
  - metrics: cmap_at_5=0.6258, f1_macro=0.4325, loss=0.1284, roc_auc_macro=0.9797
  - duration: 5247.8s

- **exp_002** [completed]
  - arch: [cnn_attention] 4-conv CNN with self-attention and SE blocks, followed by adaptive pooling and sigmoid output
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.3
  - metrics: cmap_at_5=0.4624, f1_macro=0.3105, loss=0.1836, roc_auc_macro=0.9664
  - duration: 3452.1s

- **exp_001** [completed]
  - arch: [custom_cnn] cnn_small_v1 baseline
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: cmap_at_5=0.1941, f1_macro=0.1522, loss=0.3019, roc_auc_macro=0.9168
  - duration: 2576.4s

- **exp_004** [completed]
  - arch: [cnn_gru_hybrid] 3-conv CNN front-end feeding a 2-layer GRU(256) with dropout, followed by a linear head with sigmoid output
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.4
  - metrics: cmap_at_5=0.0717, f1_macro=0.0718, loss=0.3517, roc_auc_macro=0.8639
  - duration: 2109.6s
