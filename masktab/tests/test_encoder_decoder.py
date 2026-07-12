import pytest
import torch
from masktab.models.encoder import Encoder
from masktab.models.decoder import Decoder

def test_encoder_init_invalid():
    with pytest.raises(ValueError, match="input_dim must be >= 1"):
        Encoder(input_dim=0)
    with pytest.raises(ValueError, match="hidden_dims must have at least 1 element"):
        Encoder(input_dim=10, hidden_dims=())

def test_encoder_forward_invalid_shape():
    encoder = Encoder(input_dim=10, hidden_dims=(5, 3))
    x = torch.randn(2, 5) # Wrong input dim
    with pytest.raises(RuntimeError, match="Expected input with last dimension 10, but got 5"):
        encoder(x)

def test_encoder_forward_valid():
    encoder = Encoder(input_dim=10, hidden_dims=(5, 3))
    x = torch.randn(2, 10)
    out = encoder(x)
    assert out.shape == (2, 3)
    assert encoder.get_output_dim() == 3

def test_decoder_init_invalid():
    with pytest.raises(ValueError, match="latent_dim must be >= 1"):
        Decoder(latent_dim=0, output_dim=10)
    with pytest.raises(ValueError, match="output_dim must be >= 1"):
        Decoder(latent_dim=5, output_dim=-1)
    with pytest.raises(ValueError, match="hidden_dims must have at least 1 element"):
        Decoder(latent_dim=5, output_dim=10, hidden_dims=[])

def test_decoder_forward_invalid_shape():
    decoder = Decoder(latent_dim=3, output_dim=10, hidden_dims=(5,))
    x = torch.randn(2, 5) # Wrong latent dim
    with pytest.raises(RuntimeError, match="Expected input with last dimension 3, but got 5"):
        decoder(x)

def test_decoder_forward_valid():
    decoder = Decoder(latent_dim=3, output_dim=10, hidden_dims=(5,))
    x = torch.randn(2, 3)
    out = decoder(x)
    assert out.shape == (2, 10)
