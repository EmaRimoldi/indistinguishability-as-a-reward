from __future__ import annotations

import contextlib
import os
from typing import Any


class NonFiniteGradientError(FloatingPointError):
    """Raised before an optimizer step can apply a non-finite gradient."""


def configure_stable_sdpa(torch_module=None) -> dict[str, Any]:
    """Disable the cuDNN SDPA backend that emitted NaN actor gradients.

    Flash, memory-efficient and math SDPA remain available as fallbacks.  The
    function is idempotent and returns a before/after record for the run audit.
    """

    if torch_module is None:
        import torch as torch_module

    cuda_backends = getattr(getattr(torch_module, "backends", None), "cuda", None)
    if cuda_backends is None:
        return {"configured": False, "reason": "torch.backends.cuda unavailable"}

    names = ("cudnn", "flash", "mem_efficient", "math")

    def snapshot() -> dict[str, bool | None]:
        values: dict[str, bool | None] = {}
        for name in names:
            query = getattr(cuda_backends, f"{name}_sdp_enabled", None)
            values[name] = bool(query()) if callable(query) else None
        return values

    before = snapshot()
    disable = getattr(cuda_backends, "enable_cudnn_sdp", None)
    if not callable(disable):
        raise RuntimeError("This Torch build cannot disable cuDNN SDPA")
    disable(False)
    after = snapshot()
    if after["cudnn"] is not False:
        raise RuntimeError("cuDNN SDPA remained enabled after the numerical guard")
    cudnn_backend = getattr(getattr(torch_module, "backends", None), "cudnn", None)
    cudnn_version = getattr(cudnn_backend, "version", lambda: None)()
    return {
        "configured": True,
        "policy": "disable_cudnn_keep_flash_mem_efficient_math",
        "before": before,
        "after": after,
        "torch_version": str(getattr(torch_module, "__version__", "unknown")),
        "cuda_version": str(
            getattr(getattr(torch_module, "version", None), "cuda", "unknown")
        ),
        "cudnn_version": cudnn_version,
    }


def stable_selective_log_softmax(logits, index):
    """Gather token log-probabilities with the normalization computed in FP32.

    Qwen is loaded in BF16, but the GRPO likelihood ratio and KL term are much
    less tolerant of low-precision log-softmax backward passes.  The returned
    tensor remains connected to the original logits, so this changes numerical
    precision only; it does not change the objective or sampling distribution.
    """

    import torch

    work_logits = logits
    if logits.dtype in (torch.float16, torch.bfloat16):
        work_logits = logits.float()
    selected = torch.gather(
        work_logits, dim=-1, index=index.unsqueeze(-1)
    ).squeeze(-1)
    return selected - torch.logsumexp(work_logits, dim=-1)


def install_stable_grpo_logprobs() -> dict[str, Any]:
    """Install the FP32 token-logprob kernel used by TRL's GRPO trainer.

    TRL imports ``selective_log_softmax`` into the trainer module, so replacing
    the utility function alone is insufficient.  Installation is idempotent.
    """

    import torch
    from trl.trainer import grpo_trainer

    sdpa = configure_stable_sdpa(torch)

    current = grpo_trainer.selective_log_softmax
    if getattr(current, "_iar_fp32_stable", False):
        return {
            "installed": True,
            "already_installed": True,
            "sdpa": sdpa,
        }
    stable_selective_log_softmax._iar_fp32_stable = True  # type: ignore[attr-defined]
    grpo_trainer.selective_log_softmax = stable_selective_log_softmax
    return {
        "installed": True,
        "already_installed": False,
        "replaced": f"{current.__module__}.{current.__name__}",
        "policy_logprob_compute_dtype": "float32",
        "sdpa": sdpa,
    }


def guarded_grpo_trainer_class(base_class):
    """Return a GRPOTrainer subclass that fails before a bad optimizer step.

    Set ``IAR_GRPO_DETECT_ANOMALY=1`` only for diagnostic canaries: it is much
    slower, but PyTorch then reports the backward operation that first emits a
    non-finite value.
    """

    import torch

    class GuardedGRPOTrainer(base_class):
        def training_step(self, model, inputs, num_items_in_batch=None):
            detect = os.environ.get("IAR_GRPO_DETECT_ANOMALY", "0") == "1"
            context = (
                torch.autograd.detect_anomaly(check_nan=True)
                if detect
                else contextlib.nullcontext()
            )
            with context:
                loss = super().training_step(model, inputs, num_items_in_batch)

            named_gradients = [
                (name, parameter.grad)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if named_gradients:
                checks = torch.stack(
                    [torch.isfinite(gradient).all() for _, gradient in named_gradients]
                )
                if not bool(checks.all().item()):
                    offenders = [
                        name
                        for (name, _), finite in zip(
                            named_gradients, checks.tolist(), strict=True
                        )
                        if not finite
                    ]
                    preview = ", ".join(offenders[:12])
                    suffix = "" if len(offenders) <= 12 else f" (+{len(offenders) - 12} more)"
                    raise NonFiniteGradientError(
                        "Non-finite actor gradients detected before optimizer.step: "
                        f"{preview}{suffix}"
                    )
            return loss

    GuardedGRPOTrainer.__name__ = f"FiniteGuard{base_class.__name__}"
    return GuardedGRPOTrainer
