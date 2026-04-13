### Top-K Successful Experiments
- **exp_010** [completed]
  - arch: custom 4-block CNN with global average pooling and dropout
  - hyperparams: lr=0.001, batch_size=32, epochs=5, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: loss=0.0848, roc_auc_macro=0.9334
  - duration: 7128.4s

- **exp_006** [completed]
  - arch: cnn_small_v1 baseline with longer training
  - hyperparams: lr=0.001, batch_size=32, epochs=5, optimizer=adam, weight_decay=0, dropout=0.1
  - metrics: loss=0.0221, roc_auc_macro=0.8818
  - duration: 6093.2s

- **exp_001** [completed]
  - arch: cnn_small_v1
  - hyperparams: lr=0.001, batch_size=32, epochs=2, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: loss=0.0257, roc_auc_macro=0.8126
  - duration: 2544.8s

### Recent Failures
- **exp_009** [failed]
  - arch: Custom 3-conv baseline with global average pooling and dropout
  - hyperparams: lr=0.0005, batch_size=32, epochs=8, optimizer=adam, weight_decay=0.0, dropout=0.2

- **exp_008** [failed]
  - arch: cnn_small_v1 with specAug augmentation
  - hyperparams: lr=0.001, batch_size=32, epochs=5, optimizer=adam, weight_decay=0, dropout=0.1

- **exp_007** [failed]
  - arch: ResNet-20 style 3-block CNN with SE attention
  - hyperparams: lr=0.001, batch_size=64, epochs=5, optimizer=adam, weight_decay=0.0001, dropout=0.1

- **exp_005** [failed]
  - arch: custom 3‑conv residual CNN with SE attention and attention dropout
  - hyperparams: lr=0.0005, batch_size=32, epochs=3, optimizer=adam, weight_decay=0.0001, dropout=0.1

- **exp_004** [failed]
  - arch: custom 3-conv baseline with spatial dropout and 1‑channel feature extraction
  - hyperparams: lr=0.001, batch_size=32, epochs=3, optimizer=adam, weight_decay=0.0001, dropout=0.2
