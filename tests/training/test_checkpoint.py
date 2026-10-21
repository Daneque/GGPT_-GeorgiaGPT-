import random
import numpy as np
import pytest
import torch

from ggpt.training.checkpoint import save_checkpoint, load_checkpoint
from ggpt.training.scheduler import create_scheduler
from ggpt.data.token_loader import ResumableRandomBatchSampler


def fixture_items():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    scheduler = create_scheduler(optimizer, total_steps=8, warmup_steps=2)
    sampler = ResumableRandomBatchSampler(50, 2, 12, 123)
    return model, optimizer, scheduler, sampler


def test_roundtrip_rng_scheduler_and_atomic(tmp_path):
    random.seed(19); np.random.seed(21); torch.manual_seed(23)
    model, opt, sch, sampler = fixture_items()
    x = torch.ones((2, 3))
    model(x).sum().backward(); opt.step(); sch.step(); opt.zero_grad()
    checkpoint = tmp_path / "latest.pt"
    progress = {"step": 1, "epoch": 0, "next_batch": 2, "tokens_seen": 8,
                "best_val_loss": 4.5}
    save_checkpoint(checkpoint, model=model, optimizer=opt, scheduler=sch,
                    progress=progress, sampler_state=sampler.state_dict(next_batch=2),
                    run_spec={"name": "example"})
    expected_py = random.random()
    expected_np = np.random.random()
    expected_torch = torch.rand(3)
    original_weights = [p.detach().clone() for p in model.parameters()]
    model2, opt2, sch2, sampler2 = fixture_items()
    restored = load_checkpoint(checkpoint, model=model2, optimizer=opt2,
                               scheduler=sch2, sampler=sampler2,
                               run_spec={"name": "example"})
    assert restored == progress
    for a, b in zip(original_weights, model2.parameters()):
        torch.testing.assert_close(a, b)
    assert random.random() == expected_py
    assert np.random.random() == expected_np
    torch.testing.assert_close(torch.rand(3), expected_torch)
    assert sampler2.start_batch == 2
    assert opt2.param_groups[0]["lr"] == opt.param_groups[0]["lr"]
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError, match="configuration"):
        load_checkpoint(checkpoint, model=model2, optimizer=opt2,
                        scheduler=sch2, sampler=sampler2,
                        run_spec={"name": "wrong"})

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_rng_checkpoint_restores_when_loaded_to_gpu(tmp_path):
    model, opt, sch, sampler = fixture_items()
    model = model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=0.002)
    sch = create_scheduler(opt, total_steps=8, warmup_steps=2)
    path = tmp_path / "cuda.pt"
    save_checkpoint(path, model=model, optimizer=opt, scheduler=sch,
                    progress={"step": 0}, sampler_state=sampler.state_dict(next_batch=0),
                    run_spec={"name": "cuda"})
    load_checkpoint(path, model=model, optimizer=opt, scheduler=sch, sampler=sampler,
                    run_spec={"name": "cuda"}, map_location="cuda")
    assert torch.cuda.get_rng_state().device.type == "cpu"
