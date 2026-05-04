# I-10 — TaskAdapter + BirdCLEF adapter

**Labels:** `track-task`, `p0`
**Milestone:** week-2-loop-and-task
**Owner:** Dev C

## Context
ADR-003. Task knowledge behind an adapter. Track B is the only adapter we ship.

## Scope
Define the ABC. Implement `BirdclefAdapter`. Wire `lab/tasks/registry.py` to instantiate from `settings.task.adapter` (import path `module:ClassName`).

## Interface
```python
class DatasetProfile(BaseModel):
    num_classes: int
    num_train: int
    input_tensor_shape: tuple[int, ...]     # (1,128,313)
    class_imbalance: dict[str, int] | None = None

class TaskAdapter(ABC):
    name: str
    kind: str
    primary_metric: str

    @abstractmethod
    def profile(self) -> DatasetProfile: ...
    @abstractmethod
    def prompt_slots(self) -> dict[str, str]: ...
    @abstractmethod
    def model_block_signature(self) -> tuple[str, str]: ...
    @abstractmethod
    def spawn_triggering_calls(self) -> Iterable[str]: ...
    @abstractmethod
    def build_submission(self, code: str, experiment_id: str, out_dir: Path) -> Path: ...

def get_task_adapter(settings: Settings) -> TaskAdapter: ...
```

`BirdclefAdapter.profile()` reads `data/processed/mels/metadata.parquet`:
- `num_classes` = count of unique `primary_label` (expect 234; assert equals config).
- `input_tensor_shape` = `(1, n_mels, n_frames)` from metadata.
- `class_imbalance` = per-class sample counts.

`prompt_slots()` returns:
```
{"task_description": "Multi-label bird species classification from 5s mel spectrograms.",
 "num_classes": "234",
 "input_tensor_shape": "(1, 128, 313)",
 "valid_architecture_families": "cnn_scratch, efficientnet_pretrained, mobilenet_pretrained, yamnet_feature",
 "code_skeleton_content": <rendered skeleton excerpt>}
```

## Files
- Rewrite `lab/tasks/base.py`, `lab/tasks/registry.py`, `lab/tasks/track_b_birdclef.py`
- Rewrite `config/tasks/track_b.yaml`
- Create `tests/test_track_b_adapter.py`

## Acceptance
- `profile().num_classes == 234` on real metadata parquet (asserted against the PDF spec).
- `get_task_adapter(settings)` dynamic import returns `BirdclefAdapter`.
- Submission stub returns a path but actual build lives in I-14.

## Depends on
I-01, I-02.
