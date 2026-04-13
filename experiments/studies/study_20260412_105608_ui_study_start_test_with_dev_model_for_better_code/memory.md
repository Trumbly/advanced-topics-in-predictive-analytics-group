### Top-K Successful Experiments
- **exp_003** [completed]
  - arch: [deep_cnn] 5-conv deep CNN with residual connections, BatchNorm, and SE attention blocks, followed by adaptive pooling and sigmoid output
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.3
  - metrics: loss=0.3476, roc_auc_macro=0.9385
  - duration: 1436.3s

- **exp_005** [completed]
  - arch: [custom_cnn] 4-conv from-scratch with BatchNorm, SE attention, and adaptive pooling
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.3
  - metrics: loss=0.3776, roc_auc_macro=0.9110
  - duration: 1439.3s

- **exp_006** [completed]
  - arch: [cnn_attention] 4-conv CNN with self-attention and SE blocks, followed by adaptive pooling and sigmoid output
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.3
  - metrics: loss=0.4081, roc_auc_macro=0.9073
  - duration: 748.6s

- **exp_002** [completed]
  - arch: [cnn_attention] 4-conv CNN with self-attention and SE blocks, followed by adaptive pooling and sigmoid output
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.3
  - metrics: loss=0.4039, roc_auc_macro=0.9066
  - duration: 682.0s

- **exp_001** [completed]
  - arch: [custom_cnn] cnn_small_v1 baseline
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: loss=0.4795, roc_auc_macro=0.7957
  - duration: 604.3s

### Recent Failures
- **exp_004** [failed]
  - arch: [cnn_gru_hybrid] 3-conv CNN front-end feeding a 2-layer GRU(128) with dropout, followed by a linear head with sigmoid output
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.3
