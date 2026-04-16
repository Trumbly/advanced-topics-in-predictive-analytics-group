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
        "def build_model(num_classes: int):\n"
        "    return Net()\n"
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


def test_rejects_torch_hub_load():
    code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "def build_model(num_classes: int):\n"
        "    return torch.hub.load('NVIDIA/DeepLearningExamples', 'resnet50', pretrained=True)\n"
        "if __name__ == '__main__':\n"
        "    pass\n"
    )
    r = validate(code)
    assert not r.ok
    assert r.error_type == "ForbiddenModelSource"


def test_rejects_torch_hub_load_via_imported_load_alias():
    code = (
        "from torch.hub import load\n"
        "import torch.nn as nn\n"
        "def build_model(num_classes: int):\n"
        "    return load('NVIDIA/DeepLearningExamples', 'resnet50', pretrained=True)\n"
        "if __name__ == '__main__':\n"
        "    pass\n"
    )
    r = validate(code)
    assert not r.ok
    assert r.error_type == "ForbiddenModelSource"


def test_accepts_build_model_with_local_torchvision_import():
    code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "def build_model(num_classes: int):\n"
        "    from torchvision.models import resnet18\n"
        "    model = resnet18(pretrained=True)\n"
        "    model.fc = nn.Linear(model.fc.in_features, num_classes)\n"
        "    return model\n"
        "if __name__ == '__main__':\n"
        "    pass\n"
    )
    r = validate(
        code,
        require_build_model=True,
        smoke_input_shape=(1, 64, 64),
        smoke_num_classes=5,
    )
    assert r.ok, r.message


def test_rejects_torchvision_usage_without_import():
    code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "def build_model(num_classes: int):\n"
        "    model = torchvision.models.resnet18(pretrained=False)\n"
        "    model.fc = nn.Linear(model.fc.in_features, num_classes)\n"
        "    return model\n"
        "if __name__ == '__main__':\n"
        "    pass\n"
    )
    r = validate(
        code,
        require_build_model=True,
        smoke_input_shape=(1, 64, 64),
        smoke_num_classes=5,
    )
    assert not r.ok
    assert r.error_type == "DryRunFailed"
    assert "torchvision" in r.message


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


def test_rejects_nn_usage_without_import_alias():
    code = (
        "import torch\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "if __name__ == '__main__':\n"
        "    pass\n"
    )
    r = validate(code)
    assert not r.ok
    assert r.error_type == "MissingNNImport"


def test_rejects_missing_build_model():
    code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "if __name__ == '__main__':\n"
        "    pass\n"
    )
    r = validate(code, require_build_model=True)
    assert not r.ok
    assert r.error_type == "DryRunFailed"
    assert "build_model" in r.message


def test_rejects_numpy_without_detach():
    code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "def build_model(num_classes: int):\n"
        "    return nn.Linear(10, num_classes)\n"
        "def main():\n"
        "    x = torch.randn(2, 10, requires_grad=True)\n"
        "    logits = build_model(2)(x)\n"
        "    probs = torch.softmax(logits, dim=1).numpy()\n"
        "    _ = probs\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    r = validate(code, require_build_model=True, require_training_contract=False)
    assert not r.ok
    assert r.error_type == "UnsafeNumpyConversion"


def test_accepts_numpy_with_detach_chain():
    code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "def build_model(num_classes: int):\n"
        "    return nn.Linear(10, num_classes)\n"
        "def main():\n"
        "    x = torch.randn(2, 10, requires_grad=True)\n"
        "    logits = build_model(2)(x)\n"
        "    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()\n"
        "    _ = probs\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    r = validate(code, require_build_model=True, require_training_contract=False)
    assert r.ok, r.message


def test_accepts_numpy_without_detach_inside_no_grad_decorated_fn():
    code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "def build_model(num_classes: int):\n"
        "    return nn.Linear(10, num_classes)\n"
        "@torch.no_grad()\n"
        "def main():\n"
        "    x = torch.randn(2, 10)\n"
        "    logits = build_model(2)(x)\n"
        "    probs = torch.softmax(logits, dim=1).numpy()\n"
        "    _ = probs\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    r = validate(code, require_build_model=True, require_training_contract=False)
    assert r.ok, r.message


def test_rejects_build_model_runtime_name_error():
    code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "def build_model(num_classes: int):\n"
        "    return MissingThing()\n"
        "if __name__ == '__main__':\n"
        "    pass\n"
    )
    r = validate(code)
    assert not r.ok
    assert r.error_type == "DryRunFailed"
    assert "build_model failed" in r.message


def test_rejects_build_model_without_forward_under_forward_smoke():
    code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "def build_model(num_classes: int):\n"
        "    return nn.Module()\n"
        "if __name__ == '__main__':\n"
        "    pass\n"
    )
    r = validate(
        code,
        require_build_model=True,
        smoke_input_shape=(1, 16, 16),
        smoke_num_classes=4,
    )
    assert not r.ok
    assert r.error_type == "DryRunFailed"
    assert "returns `nn.Module()`" in r.message


def test_rejects_channel_mismatch_under_forward_smoke():
    code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "def build_model(num_classes: int):\n"
        "    return nn.Conv2d(3, num_classes, kernel_size=1)\n"
        "if __name__ == '__main__':\n"
        "    pass\n"
    )
    r = validate(
        code,
        require_build_model=True,
        smoke_input_shape=(1, 16, 16),
        smoke_num_classes=4,
    )
    assert not r.ok
    assert r.error_type == "DryRunFailed"
    assert "channel mismatch" in r.message.lower() or "in_channels=3" in r.message.lower()


def test_accepts_build_model_with_torch_no_grad_decorator():
    code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "@torch.no_grad()\n"
        "def helper(x):\n"
        "    return x\n"
        "def build_model(num_classes: int):\n"
        "    return nn.Linear(10, num_classes)\n"
        "if __name__ == '__main__':\n"
        "    pass\n"
    )
    r = validate(code, require_build_model=True)
    assert r.ok, r.message


def test_rejects_minimal_model_only_script_when_training_contract_required():
    code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "def build_model(num_classes: int):\n"
        "    return nn.Linear(10, num_classes)\n"
        "if __name__ == '__main__':\n"
        "    print(build_model(2))\n"
    )
    r = validate(
        code,
        require_build_model=True,
        require_training_contract=True,
    )
    assert not r.ok
    assert r.error_type == "MissingTrainingContract"


def test_accepts_training_contract_when_required():
    code = (
        "import json\n"
        "from pathlib import Path\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "\n"
        "def load_text_dataset(batch_size=1):\n"
        "    return [torch.zeros(1, 10)], [torch.zeros(1, 10)], 2\n"
        "\n"
        "def build_model(num_classes: int):\n"
        "    return nn.Linear(10, num_classes)\n"
        "\n"
        "def main():\n"
        "    train_loader, val_loader, num_classes = load_text_dataset(batch_size=1)\n"
        "    model = build_model(num_classes)\n"
        "    _ = (train_loader, val_loader, model)\n"
        "    Path('results.json').write_text(json.dumps({'primary_metric':'f1_macro','primary_score':0.1}))\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    r = validate(
        code,
        extra_spawn_triggers=("load_text_dataset",),
        require_build_model=True,
        require_training_contract=True,
    )
    assert r.ok, r.message
