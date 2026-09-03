import ast
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_training_step():
    """Load the pure training function without importing GPU-only modules."""
    source_path = Path(__file__).parents[1] / "train.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "training_step"
    )
    namespace = {"torch": torch, "F": F, "CrossEntropyLoss": torch.nn.CrossEntropyLoss}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    return namespace["training_step"]


training_step = _load_training_step()


class _SourceDistribution:
    def sample_like(self, x):
        return torch.zeros_like(x)


class _PathSample:
    def __init__(self, x_t):
        self.x_t = x_t


class _Path:
    def sample(self, x_0, x_1, t):
        del x_0, t
        return _PathSample(x_1)


class _Model:
    dtype = torch.float32

    def __init__(self, logits):
        self.logits = logits

    def __call__(self, x_t, data_info):
        del x_t, data_info
        return None, self.logits


class _Args:
    l2_loss_weight = 0


class _NumericProcessor:
    num_start_id = 10
    num_end_id = 12
    min_num = 0.0
    max_num = 2.0
    interval = 1.0


def _batch(logits, targets, text_token_mask):
    return {
        "text_token_mask": text_token_mask,
        "understanding_img": torch.zeros(1),
        "input_ids": targets,
    }


def test_training_step_supports_variable_text_lengths():
    # Sample 0 supervises two positions while sample 1 supervises one.  The
    # flattened selection must preserve all three target/logit pairs.
    targets = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]])
    text_token_mask = torch.tensor(
        [[True, True, False, False], [True, False, False, False]]
    )
    logits = torch.full((2, 4, 5), -4.0)
    logits[0, 0, 1] = 4.0
    logits[0, 1, 2] = 4.0
    logits[1, 0, 3] = 4.0

    loss, logs = training_step(
        model=_Model(logits),
        x_1=targets,
        source_distribution=_SourceDistribution(),
        data_info=_batch(logits, targets, text_token_mask),
        path=_Path(),
        stage="s1",
        args=_Args(),
    )

    assert torch.isfinite(loss)
    assert logs["ce_loss"] < 0.01


def test_stage_two_without_numeric_tokenizer_does_not_enter_numeric_loss():
    targets = torch.tensor([[1, 2]])
    text_token_mask = torch.ones_like(targets, dtype=torch.bool)
    logits = torch.zeros(1, 2, 4)

    loss, logs = training_step(
        model=_Model(logits),
        x_1=targets,
        source_distribution=_SourceDistribution(),
        data_info=_batch(logits, targets, text_token_mask),
        path=_Path(),
        stage="s2",
        vl_chat_processor=object(),
        args=_Args(),
    )

    assert torch.isfinite(loss)
    assert "l2_loss" not in logs


def test_default_cross_entropy_loss_is_callable():
    targets = torch.tensor([[1]])
    logits = torch.zeros(1, 1, 2)

    loss, _ = training_step(
        model=_Model(logits),
        x_1=targets,
        source_distribution=_SourceDistribution(),
        data_info=_batch(logits, targets, torch.ones_like(targets, dtype=torch.bool)),
        path=_Path(),
        stage="s1",
        args=_Args(),
    )

    assert torch.isfinite(loss)


def test_numeric_auxiliary_loss_uses_configured_weight():
    class Args:
        l2_loss_weight = 2.0

    targets = torch.tensor([[10, 11]])
    logits = torch.zeros(1, 2, 12)
    logits[:, :, 11] = 1.0

    loss, logs = training_step(
        model=_Model(logits),
        x_1=targets,
        source_distribution=_SourceDistribution(),
        data_info=_batch(logits, targets, torch.ones_like(targets, dtype=torch.bool)),
        path=_Path(),
        stage="s2",
        vl_chat_processor=_NumericProcessor(),
        args=Args(),
    )

    # Predicted numeric values are [1, 1], targets are [0, 1], so MSE=0.5 and
    # the configured coefficient contributes exactly 1.0 to the loss.
    assert logs["l2_loss"] == 1.0
    assert torch.isfinite(loss)
