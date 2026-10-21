import pytest
import torch

from ggpt.training.scheduler import create_scheduler, lr_multiplier


def test_schedule_boundaries_and_monotonic_cosine():
    fn = lambda step: lr_multiplier(step, total_steps=10, warmup_steps=3, min_lr_ratio=0.1)
    assert fn(0) == pytest.approx(1 / 3)
    assert fn(1) == pytest.approx(2 / 3)
    assert fn(2) == pytest.approx(1)
    assert fn(3) == pytest.approx(1)
    assert fn(10) == pytest.approx(0.1)
    assert all(fn(i) >= fn(i + 1) for i in range(3, 12))


def test_no_warmup_and_invalid_args():
    assert lr_multiplier(0, total_steps=5, warmup_steps=0, min_lr_ratio=0.2) == 1
    for kwargs in ({"warmup_steps": 5}, {"warmup_steps": -1}, {"total_steps": 0},
                   {"min_lr_ratio": -0.1}, {"min_lr_ratio": float("nan")}):
        args = dict(total_steps=5, warmup_steps=2, min_lr_ratio=0.1)
        args.update(kwargs)
        with pytest.raises(ValueError):
            lr_multiplier(0, **args)


def test_scheduler_step_order_and_restore():
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003)
    sch = create_scheduler(opt, total_steps=10, warmup_steps=3, min_lr_ratio=0.1)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.001)
    for _ in range(4):
        opt.step()
        sch.step()
    state_opt, state_sch = opt.state_dict(), sch.state_dict()
    model2 = torch.nn.Linear(2, 2)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=0.003)
    sch2 = create_scheduler(opt2, total_steps=10, warmup_steps=3, min_lr_ratio=0.1)
    opt2.load_state_dict(state_opt)
    sch2.load_state_dict(state_sch)
    assert opt2.param_groups[0]["lr"] == pytest.approx(opt.param_groups[0]["lr"])
    for _ in range(5):
        opt.step(); sch.step()
        opt2.step(); sch2.step()
        assert opt2.param_groups[0]["lr"] == pytest.approx(opt.param_groups[0]["lr"])
