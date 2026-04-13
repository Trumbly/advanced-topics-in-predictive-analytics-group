"""Unit tests for `pipelines.models`.

The model classes in this module are imported by LLM-generated code via
the registry's import_snippet strings, so these tests verify the exact
contract: import path, class name, constructor signature, forward shape.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from pipelines.models import CnnSmallV1, TorchvisionAdapter


class TestCnnSmallV1:
    def test_import_path_matches_registry(self) -> None:
        """Regression: the class name + import path must match the registry's
        import_snippet exactly, because LLM code uses that snippet verbatim."""
        from pipelines.models import CnnSmallV1 as Reimported
        assert Reimported is CnnSmallV1

    def test_default_construction(self) -> None:
        model = CnnSmallV1()
        assert model is not None

    def test_forward_matches_registry_shape(self) -> None:
        """Registry says input_shape: [1, 128, 256]. Verify that shape works."""
        model = CnnSmallV1(num_classes=234, in_channels=1)
        x = torch.randn(2, 1, 128, 256)
        out = model(x)
        assert out.shape == (2, 234)

    def test_forward_any_spatial_size(self) -> None:
        """AdaptiveAvgPool2d should accept any spatial resolution."""
        model = CnnSmallV1(num_classes=10, in_channels=1)
        x = torch.randn(3, 1, 64, 64)
        out = model(x)
        assert out.shape == (3, 10)

    def test_multichannel_input(self) -> None:
        model = CnnSmallV1(num_classes=5, in_channels=3)
        x = torch.randn(1, 3, 128, 128)
        out = model(x)
        assert out.shape == (1, 5)

    def test_parameter_count_reasonable(self) -> None:
        """Should be a small, fast baseline. Not a hard bound — just sanity."""
        model = CnnSmallV1(num_classes=234)
        total = sum(p.numel() for p in model.parameters())
        assert 5_000 < total < 200_000, f"Unexpected size: {total}"

    def test_output_is_logits_not_probabilities(self) -> None:
        """Output must be raw logits so BCEWithLogitsLoss works directly.
        Check by confirming some values fall outside [0, 1]."""
        torch.manual_seed(0)
        model = CnnSmallV1(num_classes=234)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(4, 1, 128, 128))
        # At least one value outside [0, 1] confirms these are not sigmoided
        assert (out < 0).any() or (out > 1).any() or out.abs().max() > 1


class TestTorchvisionAdapter:
    """TorchvisionAdapter wraps torchvision classifiers so our (B, 1, 128, T)
    spectrograms can feed into them without the LLM reinventing the
    1→3-channel conversion and 224×224 resize."""

    def test_efficientnet_b0_real_input_shape(self) -> None:
        """Registry says input_shape: [1, 128, 313]. Verify it works."""
        model = TorchvisionAdapter("efficientnet_b0", num_classes=206)
        x = torch.randn(2, 1, 128, 313)
        out = model(x)
        assert out.shape == (2, 206)

    def test_resnet18_real_input_shape(self) -> None:
        model = TorchvisionAdapter("resnet18", num_classes=206)
        x = torch.randn(2, 1, 128, 313)
        out = model(x)
        assert out.shape == (2, 206)

    def test_mobilenet_v3_small_real_input_shape(self) -> None:
        model = TorchvisionAdapter("mobilenet_v3_small", num_classes=206)
        x = torch.randn(2, 1, 128, 313)
        out = model(x)
        assert out.shape == (2, 206)

    def test_pretrained_false_does_not_fetch(self) -> None:
        """With pretrained=False, construction must NOT hit the internet."""
        # If this test hangs on network, something is wrong with our
        # torchvision call signature.
        model = TorchvisionAdapter(
            "efficientnet_b0", num_classes=10, pretrained=False
        )
        assert model is not None

    def test_unknown_backbone_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown torchvision backbone"):
            TorchvisionAdapter("not_a_real_model", num_classes=10)

    def test_accepts_3channel_input(self) -> None:
        """When the LLM accidentally already produces 3-channel input,
        the adapter must pass it through (no channel expand)."""
        model = TorchvisionAdapter("mobilenet_v3_small", num_classes=5)
        x = torch.randn(2, 3, 128, 313)
        out = model(x)
        assert out.shape == (2, 5)

    def test_output_is_logits(self) -> None:
        model = TorchvisionAdapter("mobilenet_v3_small", num_classes=10)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, 1, 128, 313))
        assert (out < 0).any() or (out > 1).any() or out.abs().max() > 1

    def test_device_placement_roundtrip(self) -> None:
        """LazyLinear materializes on first forward; it must end up on the
        same device as the rest of the model (CPU path — MPS/CUDA tested
        manually on the M4)."""
        model = TorchvisionAdapter("mobilenet_v3_small", num_classes=10)
        model = model.to("cpu")
        x = torch.randn(1, 1, 128, 313)
        out = model(x)
        assert out.device.type == "cpu"
        assert out.shape == (1, 10)
