import pickle
import numpy as np
import pytest
import torch
from ggpt.data.token_dataset import TokenWindowDataset
from ggpt.data.token_loader import LoaderConfig, create_dataloaders


def write_tokens(path, values):
    np.asarray(values, dtype='<u2').tofile(path)
    return path


def test_window_shift_and_boundaries(tmp_path):
    path = write_tokens(tmp_path / 'train.bin', range(12))
    dataset = TokenWindowDataset(path, context_length=4, vocab_size=32)
    assert len(dataset) == 8
    assert dataset._mapping is None
    for index in (0, 7):
        batch = dataset[index]
        assert batch['input_ids'].dtype == torch.long
        assert batch['targets'].dtype == torch.long
        assert batch['input_ids'].tolist() == list(range(index, index + 4))
        assert batch['targets'].tolist() == list(range(index + 1, index + 5))
    with pytest.raises(IndexError):
        dataset[-1]
    with pytest.raises(IndexError):
        dataset[8]
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._mapping is None
    assert torch.equal(restored[0]['targets'], dataset[0]['targets'])
    dataset[0]['input_ids'][0] = 20
    assert dataset[0]['input_ids'][0].item() == 0


def test_little_endian(tmp_path):
    path = tmp_path / 'raw.bin'
    path.write_bytes(b'\x01\x00\x02\x01\x03\x02')
    item = TokenWindowDataset(path, 2, 1024)[0]
    assert item['targets'].tolist() == [258, 515]


def test_invalid_files_and_tokens(tmp_path):
    with pytest.raises(FileNotFoundError):
        TokenWindowDataset(tmp_path / 'missing')
    odd = tmp_path / 'odd.bin'
    odd.write_bytes(b'abc')
    with pytest.raises(ValueError):
        TokenWindowDataset(odd, 1)
    short = write_tokens(tmp_path / 'short.bin', [1, 2])
    with pytest.raises(ValueError):
        TokenWindowDataset(short, 2)
    bad = write_tokens(tmp_path / 'bad.bin', [1, 2, 32])
    with pytest.raises(ValueError):
        TokenWindowDataset(bad, 2, 32)[0]
    dataset = TokenWindowDataset(short, 1)
    short.write_bytes(b'')
    with pytest.raises(RuntimeError):
        dataset[0]


def test_validation_windows(tmp_path):
    path = write_tokens(tmp_path / 'val.bin', range(20))
    first = TokenWindowDataset(path, 4, 32, validation_windows=4)
    second = TokenWindowDataset(path, 4, 32, validation_windows=4)
    assert [first.window_start(i) for i in range(4)] == [0, 5, 10, 15]
    for i in range(4):
        assert torch.equal(first[i]['input_ids'], second[i]['input_ids'])
    assert len(TokenWindowDataset(path, 4, 32, validation_windows=100)) == 16


def collect(loader):
    return torch.cat([batch['input_ids'] for batch in loader])


@pytest.mark.parametrize('workers', [0, 2])
def test_loaders_reproducible_and_isolated(tmp_path, workers):
    train = write_tokens(tmp_path / 'train.bin', range(40))
    val = write_tokens(tmp_path / 'val.bin', range(50, 90))
    config = LoaderConfig(context_length=4, batch_size=3, train_windows=17,
                          validation_windows=6, vocab_size=128, num_workers=workers)
    a, av = create_dataloaders(config, train, val)
    b, bv = create_dataloaders(config, train, val)
    train_a, train_b = collect(a), collect(b)
    assert torch.equal(train_a, train_b)
    assert train_a.shape == (17, 4)
    assert train_a.max() < 40
    validation = collect(av)
    assert validation.min() >= 50
    assert torch.equal(validation, collect(av))
    assert torch.equal(validation, collect(bv))
    with pytest.raises(ValueError):
        create_dataloaders(config, train, train)


def test_worker_count_does_not_change_sampling(tmp_path):
    train = write_tokens(tmp_path / 'train.bin', range(40))
    val = write_tokens(tmp_path / 'val.bin', range(50, 90))
    options = dict(context_length=4, train_windows=20, vocab_size=128)
    a, _ = create_dataloaders(LoaderConfig(**options, num_workers=0), train, val)
    b, _ = create_dataloaders(LoaderConfig(**options, num_workers=2), train, val)
    assert torch.equal(collect(a), collect(b))


@pytest.mark.parametrize('workers', [0, 2])
def test_batch_sampler_checkpoint_exact_resume_with_prefetch(tmp_path, workers):
    train = write_tokens(tmp_path / 'train.bin', range(240))
    val = write_tokens(tmp_path / 'val.bin', range(240, 480))
    config = LoaderConfig(context_length=4, batch_size=3, train_windows=20,
                          validation_windows=6, vocab_size=512, num_workers=workers, seed=919)
    entire, _ = create_dataloaders(config, train, val)
    all_batches = [batch['input_ids'].clone() for batch in entire]
    assert len(all_batches) == 7  # The final batch has only two windows.
    assert all_batches[-1].shape[0] == 2

    interrupted, _ = create_dataloaders(config, train, val)
    iterator = iter(interrupted)
    for i in range(3):
        torch.testing.assert_close(next(iterator)['input_ids'], all_batches[i])
    # Worker(s) may have prefetched more than three batches. Only three were consumed.
    state = interrupted.batch_sampler.state_dict(next_batch=3)
    del iterator

    recovered, _ = create_dataloaders(config, train, val)
    recovered.batch_sampler.load_state_dict(state)
    assert len(recovered) == 4
    actual = [batch['input_ids'] for batch in recovered]
    for observed, expected in zip(actual, all_batches[3:], strict=True):
        torch.testing.assert_close(observed, expected)

    # New logical pass must not silently reuse exactly the same windows.
    recovered.batch_sampler.set_epoch(1)
    assert len(recovered) == 7
    assert not torch.equal(collect(recovered), torch.cat(all_batches))


def test_batch_sampler_state_rejects_mismatch(tmp_path):
    from ggpt.data.token_loader import ResumableRandomBatchSampler

    s = ResumableRandomBatchSampler(dataset_length=100, batch_size=4,
                                    num_samples=11, seed=12)
    assert len(s) == 3
    assert len(list(iter(s))[-1]) == 3
    with pytest.raises(TypeError):
        s.state_dict()
    with pytest.raises(ValueError):
        s.state_dict(next_batch=4)
    state = s.state_dict(next_batch=2)
    s.load_state_dict(state)
    assert len(s) == 1
    with pytest.raises(ValueError):
        s.state_dict(next_batch=1)
    with pytest.raises(ValueError):
        s.load_state_dict({**state, 'batch_size': 3})
    with pytest.raises(ValueError):
        s.load_state_dict({**state, 'epoch': -1})
    with pytest.raises(ValueError):
        s.load_state_dict({**state, 'next_batch': 4})
    with pytest.raises(ValueError):
        s.set_epoch(-1)
    with pytest.raises(ValueError):
        s.set_start_batch(-1)

