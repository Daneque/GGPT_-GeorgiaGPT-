import json
from pathlib import Path

import pytest
import torch

from ggpt.inference.artifact import export_model, file_sha256
from ggpt.model import ModelConfig, create_model


def test_export_excludes_training_state(tmp_path):
    config = ModelConfig(vocab_size=16, hidden_size=8, num_heads=2,
                         num_layers=1, intermediate_size=16, max_seq_len=16)
    model = create_model(config)
    tokenizer = tmp_path / 'tokenizer.json'
    tokenizer.write_text('{"fake":"test"}')
    run = tmp_path / 'training_run'
    (run / 'checkpoints').mkdir(parents=True)
    spec = {'model': config.to_dict(), 'tokenizer': {'path': str(tokenizer)}}
    (run / 'run_spec.json').write_text(json.dumps(spec))
    torch.save({
        'run_spec': spec, 'model': model.state_dict(), 'progress': {'step': 400000},
        'optimizer': {'fake': torch.ones(3)}, 'scheduler': {'fake': 'not-exported'},
    }, run / 'checkpoints/latest.pt')
    destination = tmp_path / 'standalone_model'
    export_model(run, destination)
    assert sorted(f.name for f in destination.iterdir()) == [
        'manifest.json', 'model.pt', 'tokenizer.json'
    ]
    state = torch.load(destination / 'model.pt', weights_only=True)
    assert state.keys() == model.state_dict().keys()
    assert 'optimizer' not in state
    manifest = json.loads((destination / 'manifest.json').read_text())
    assert manifest['step'] == 400000
    assert manifest['files']['model.pt'] == file_sha256(destination / 'model.pt')
    with pytest.raises(FileExistsError):
        export_model(run, destination)
