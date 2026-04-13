### Top-K Successful Experiments
- **exp_007** [completed]
  - arch: custom 3-conv CNN with fused residual blocks and SE attention
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.1
  - metrics: loss=0.0382, roc_auc_macro=0.5613
  - duration: 755.0s

- **exp_008** [completed]
  - arch: cnn_small_v1 | pretrained: cnn_small_v1
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.1
  - metrics: loss=0.1028, roc_auc_macro=0.5447
  - duration: 100.4s

- **exp_004** [completed]
  - arch: 4-layer ResNet block with SE attention
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.1
  - metrics: loss=0.0921, roc_auc_macro=0.5286
  - duration: 782.2s

- **exp_020** [completed]
  - arch: Enhanced 3-conv CNN with residual SE attention and dropout
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.2
  - metrics: loss=0.0559, roc_auc_macro=0.5264
  - duration: 944.5s

- **exp_018** [completed]
  - arch: cnn_small_v1 baseline with dropout | pretrained: cnn_small_v1
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.2
  - metrics: loss=0.1050, roc_auc_macro=0.5253
  - duration: 0.0s

- **exp_005** [completed]
  - arch: custom 3-conv CNN with dropout
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.2
  - metrics: loss=0.0417, roc_auc_macro=0.5144
  - duration: 776.4s

- **exp_001** [completed]
  - arch: cnn_small_v1
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: loss=0.1055, roc_auc_macro=0.5081
  - duration: 128.4s

- **exp_014** [completed]
  - arch: 3-conv CNN with fused residual blocks, SE attention, and mixup augmentation
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.1
  - metrics: loss=0.0428, roc_auc_macro=0.4938
  - duration: 755.5s

- **exp_006** [completed]
  - arch: custom 3-conv CNN with SE attention and dropout
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.2
  - metrics: loss=0.0847, roc_auc_macro=0.4882
  - duration: 964.8s

### Recent Failures
- **exp_025** [failed]
  - arch: 3-conv CNN with fused SE attention and dropout
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.1

- **exp_024** [failed]
  - arch: 3-conv baseline with fused SE attention and dropout | pretrained: cnn_small_v1
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.2

- **exp_023** [failed]

- **exp_022** [failed]
  - arch: 3‑conv concatenated CNN with residual SE attention and dropout
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.1

- **exp_021** [failed]
  - arch: ResNet-style 4-block CNN with fused SE attention and adaptive dropout
  - hyperparams: lr=0.0005, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.4

- **exp_019** [failed]
  - arch: 3-conv CNN with fused residual blocks and SE attention
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.1

- **exp_017** [failed]
  - arch: 3-conv CNN with fused residual blocks and SE attention
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.1

- **exp_016** [failed]
  - arch: EfficientNet-B0 full model with linear head | pretrained: efficientnet_b0
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.1

- **exp_015** [failed]
  - arch: custom 3-conv CNN with fused residual blocks, SE attention, dropout, and mixup augmentation
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.2

- **exp_013** [failed]
  - arch: EfficientNet-B0 truncated to first two blocks + linear head | pretrained: efficientnet_b0
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.1

- **exp_012** [failed]
  - arch: custom 3-conv CNN with fused residual blocks, SE attention, and dropout
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.1

- **exp_011** [failed]
  - arch: EfficientNet-B0 backbone with linear output head | pretrained: efficientnet_b0
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.1
