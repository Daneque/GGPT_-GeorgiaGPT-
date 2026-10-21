import pytest
import torch

from ggpt.model import ModelConfig, create_model
from ggpt.training import (
    OptimizerConfig, StepConfig, benchmark_train_steps, create_optimizer,
)


def setup_benchmark():
    model = create_model(ModelConfig(vocab_size=32, hidden_size=16, num_heads=2,
                                     intermediate_size=24, num_layers=1, max_seq_len=5))
    optimizer = create_optimizer(model, OptimizerConfig(fused=False))
    x = torch.randint(0, 32, (1, 4))
    group = [{'input_ids': x, 'targets': x}]
    return model, optimizer, [group, group, group]


def test_benchmark_cpu_reports_steps_tokens_and_time():
    model, optimizer, groups = setup_benchmark()
    result = benchmark_train_steps(model, optimizer, groups, 'cpu', StepConfig(),
                                   warmup_steps=1, measured_steps=2)
    assert result.warmup_steps == 1 and result.measured_steps == 2
    assert result.tokens == 8
    assert result.mean_step_seconds > 0 and result.median_step_seconds > 0
    assert result.tokens_per_second > 0
    assert result.peak_allocated_mib is None and result.peak_reserved_mib is None


def test_benchmark_validates_number_of_groups():
    model, optimizer, groups = setup_benchmark()
    with pytest.raises(ValueError, match='expected'):
        benchmark_train_steps(model, optimizer, groups[:2], 'cpu', StepConfig(),
                              warmup_steps=1, measured_steps=2)
    with pytest.raises(ValueError, match='measured_steps'):
        benchmark_train_steps(model, optimizer, groups, 'cpu', StepConfig(),
                              warmup_steps=1, measured_steps=0)
