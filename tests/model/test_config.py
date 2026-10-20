import pytest
from ggpt.model import ModelConfig, create_model


def test_json_roundtrip():
    config = ModelConfig(dropout=0.1, max_seq_len=128)
    assert ModelConfig.from_json(config.to_json()) == config
    assert ModelConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize('changes', [
    {'hidden_size': 513}, {'hidden_size': 24, 'num_heads': 8},
    {'num_layers': 0}, {'max_seq_len': -1}, {'vocab_size': True},
    {'dropout': 1}, {'dropout': float('nan')}, {'rope_theta': 0},
    {'rms_norm_eps': -1}, {'initializer_range': 0}, {'num_heads': 0},
    {'bos_token_id': 1}, {'pad_token_id': 20000}, {'num_layers': 2.0},
])
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        ModelConfig(**changes)


def test_unknown_fields_and_factory():
    with pytest.raises(TypeError):
        ModelConfig.from_json('{"unknown": 1}')
    with pytest.raises(ValueError):
        ModelConfig.from_json('[]')
    with pytest.raises(ValueError, match='Unknown'):
        create_model(ModelConfig(model_type='unsupported'))

