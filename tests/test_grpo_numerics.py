import pytest

torch = pytest.importorskip("torch")

from iar_grpo.numerics import (
    NonFiniteGradientError,
    configure_stable_sdpa,
    guarded_grpo_trainer_class,
    stable_selective_log_softmax,
)


def test_stable_sdpa_disables_only_cudnn_and_is_idempotent():
    class FakeCudaBackends:
        def __init__(self):
            self.cudnn = True

        def cudnn_sdp_enabled(self):
            return self.cudnn

        def flash_sdp_enabled(self):
            return True

        def mem_efficient_sdp_enabled(self):
            return True

        def math_sdp_enabled(self):
            return True

        def enable_cudnn_sdp(self, value):
            self.cudnn = value

    class FakeCudnn:
        @staticmethod
        def version():
            return 92000

    class FakeTorch:
        __version__ = "test"
        version = type("Version", (), {"cuda": "test"})()
        backends = type(
            "Backends",
            (),
            {"cuda": FakeCudaBackends(), "cudnn": FakeCudnn()},
        )()

    first = configure_stable_sdpa(FakeTorch)
    second = configure_stable_sdpa(FakeTorch)
    assert first["before"]["cudnn"] is True
    assert first["after"] == {
        "cudnn": False,
        "flash": True,
        "mem_efficient": True,
        "math": True,
    }
    assert second["before"]["cudnn"] is False
    assert second["after"]["cudnn"] is False


def test_stable_selective_log_softmax_matches_fp32_and_has_finite_gradient():
    logits = torch.tensor(
        [[[80.0, -80.0, 0.0], [-70.0, 70.0, 0.0]]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    index = torch.tensor([[0, 1]])
    actual = stable_selective_log_softmax(logits, index)
    reference = torch.gather(
        logits.float().log_softmax(-1), -1, index.unsqueeze(-1)
    ).squeeze(-1)
    assert actual.dtype == torch.float32
    assert torch.allclose(actual, reference)
    (-actual.mean()).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_gradient_guard_rejects_nan_before_returning_training_step():
    class BaseTrainer:
        def training_step(self, model, inputs, num_items_in_batch=None):
            del inputs, num_items_in_batch
            parameter = next(model.parameters())
            parameter.grad = torch.full_like(parameter, float("nan"))
            return torch.tensor(0.0)

    trainer = guarded_grpo_trainer_class(BaseTrainer)()
    model = torch.nn.Linear(1, 1, bias=False)
    with pytest.raises(NonFiniteGradientError, match="before optimizer.step"):
        trainer.training_step(model, {})


def test_gradient_guard_allows_finite_gradients():
    class BaseTrainer:
        def training_step(self, model, inputs, num_items_in_batch=None):
            del inputs, num_items_in_batch
            parameter = next(model.parameters())
            parameter.grad = torch.full_like(parameter, 0.25)
            return torch.tensor(1.0)

    trainer = guarded_grpo_trainer_class(BaseTrainer)()
    model = torch.nn.Linear(1, 1, bias=False)
    assert trainer.training_step(model, {}).item() == 1.0
