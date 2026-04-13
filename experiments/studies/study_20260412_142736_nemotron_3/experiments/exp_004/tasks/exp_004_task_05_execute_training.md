# Task exp_004_task_05_execute_training

- **Experiment:** exp_004
- **Type:** predefined
- **Name:** execute_training
- **Status:** failed
- **Started:** 2026-04-12 12:41:24.556735+00:00
- **Completed:** 2026-04-12 12:41:59.559998+00:00

## Code Used
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

class MyDataset(data.Dataset):
    def __init__(self, transform=None):
        self.dataset = datasets.MNIST(root='.', train=True, download=True, transform=transform)
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform: img = self.transform(img)
        return img, label

def train():
    model = MyModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.01)
    train_set = MyDataset()
    loader = data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    for epoch in range(3):
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    train()

```

## Output
- **exit_code:** 1
- **duration_seconds:** 35.00260066695046
- **workdir:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_142736_nemotron_3/exp_004
- **results_json_path:** None
- **timed_out:** False

## Error
- **type:** UnknownError
- **message:** TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>

```
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 741, in __next__
    data = self._next_data()
           ^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 1548, in _next_data
    return self._process_data(data, worker_id)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 1586, in _process_data
    data.reraise()
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/_utils.py", line 785, in reraise
    raise exception
TypeError: Caught TypeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/utils/data/_utils/worker.py", line 358, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/utils/data/_utils/fetch.py", line 57, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py", line 401, in default_collate
    return collate(batch, collate_fn_map=default_collate_fn_map)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py", line 214, in collate
    return [
           ^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py", line 215, in <listcomp>
    collate(samples, collate_fn_map=collate_fn_map)
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py", line 243, in collate
    raise TypeError(default_collate_err_msg_format.format(elem_type))
TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>

```
