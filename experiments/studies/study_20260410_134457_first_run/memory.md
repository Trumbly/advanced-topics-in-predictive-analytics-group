### Recent Failures
- **exp_005** [failed]
  - arch: cnn_small_v1 with added 1x1 convolution to accommodate 313 time frames | pretrained: cnn_small_v1
  - hyperparams: lr=0.0001, batch_size=64, epochs=30, optimizer=Adam, weight_decay=0.0001, dropout=0.1

- **exp_004** [failed]
  - arch: lightweight 2-conv CNN with per-channel SE attention and global average pooling
  - hyperparams: lr=0.0005, batch_size=128, epochs=40, optimizer=Adam, weight_decay=0.0001, dropout=0.1
