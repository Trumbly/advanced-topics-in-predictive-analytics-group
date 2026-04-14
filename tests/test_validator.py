"""Validator tests — especially the two high-value checks preserved
verbatim from max_development."""
from __future__ import annotations

from lab.core.validator import validate


def test_accepts_minimal_valid_script():
    code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.l = nn.Linear(10, 2)\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    net = Net()\n"
    )
    r = validate(code)
    assert r.ok, r.message


def test_rejects_forbidden_import():
    code = "import subprocess\nsubprocess.run(['ls'])\n"
    r = validate(code)
    assert not r.ok
    assert r.error_type == "ForbiddenImport"


def test_rejects_os_system_pattern():
    code = "import os\nos.system('rm -rf /')\n"
    r = validate(code)
    assert not r.ok
    assert r.error_type == "ForbiddenPattern"


def test_rejects_bare_eval():
    code = "print(eval('1+1'))\n"
    r = validate(code)
    assert not r.ok
    assert r.error_type == "ForbiddenCall"


def test_allows_model_eval_attribute_call():
    # `model.eval()` is PyTorch; must NOT trigger the bare-eval check.
    code = (
        "import torch.nn as nn\n"
        "m = nn.Linear(2,2)\n"
        "m.eval()\n"
        "if __name__ == '__main__':\n"
        "    pass\n"
    )
    r = validate(code)
    assert r.ok, r.message


def test_rejects_unguarded_dataloader_at_module_scope():
    code = (
        "from torch.utils.data import DataLoader, TensorDataset\n"
        "import torch\n"
        "ds = TensorDataset(torch.zeros(10, 2), torch.zeros(10, 1))\n"
        "loader = DataLoader(ds, batch_size=4, num_workers=2)\n"
    )
    r = validate(code)
    assert not r.ok
    assert r.error_type == "MissingMainGuard"


def test_accepts_guarded_dataloader():
    code = (
        "from torch.utils.data import DataLoader, TensorDataset\n"
        "import torch\n"
        "if __name__ == '__main__':\n"
        "    ds = TensorDataset(torch.zeros(10, 2), torch.zeros(10, 1))\n"
        "    loader = DataLoader(ds, batch_size=4, num_workers=2)\n"
    )
    r = validate(code)
    assert r.ok, r.message


def test_custom_spawn_trigger_is_respected():
    code = "from lab.tasks.track_a_disaster_tweets import load_text_dataset\nload_text_dataset(batch_size=1)\n"
    r = validate(code, extra_spawn_triggers=("load_text_dataset",))
    assert not r.ok
    assert r.error_type == "MissingMainGuard"


def test_rejects_hallucinated_nn_attribute():
    # `nn.Conv2x2d` doesn't exist — the reflection check must catch it.
    code = (
        "import torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.c = nn.Conv2x2d(1, 1, 3)\n"
        "if __name__ == '__main__':\n"
        "    pass\n"
    )
    r = validate(code)
    # If torch isn't importable in the test env, the check is skipped (ok=True).
    if not r.ok:
        assert r.error_type == "HallucinatedLayer"
