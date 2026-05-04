# I-20 — Agent memory toggle + transfer-learning gate

**Labels:** `track-loop`, `p1`
**Milestone:** week-3-integration
**Owner:** Dev B

## Context
ADR-007. Optional cross-study memory. Default off. When on, transfer learning via warm-start is allowed.

## Scope
Wire `settings.agent.memory_enabled` end-to-end.

## Behavior
- **Off (default):**
  - Memory starts empty each study (apart from in-study additions).
  - `{allow_transfer_learning}` slot = `"false"` in `propose_architecture` prompt.
  - `Proposal.init_from_experiment_id` is ignored by `StudyRunner`.
  - `AGENT_CHECKPOINT_IN` env var not set on executor.
- **On:**
  - `Memory.seed_from_agent_memory(studies_root, task)` at study start.
  - `{allow_transfer_learning}` slot = `"true"`.
  - If proposal has `init_from_experiment_id`, resolve to its `checkpoint_path`; set `AGENT_CHECKPOINT_IN`.
  - If referenced experiment has no checkpoint, log warning + proceed cold.

## Interface
```python
# lab/core/lifecycle.py (extend)
class StudyRunner:
    def _wire_agent_memory(self, study: Study) -> None: ...
    def _resolve_warm_start(self, proposal: Proposal) -> str | None: ...
```

## Files
- Extend `lab/core/memory.py` (if not already in I-05)
- Extend `lab/core/lifecycle.py`
- Create `tests/test_agent_memory_toggle.py`

## Acceptance
- Off: run two studies, second gets empty memory, no warm-start env var.
- On: second study seeds memory from first; warm-start env var set when proposal requests it; not set when proposal omits it.
- Warning logged when checkpoint missing.

## Depends on
I-05, I-06.
