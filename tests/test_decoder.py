import torch

from src.models.decoder import Decoder


def create_decoder() -> Decoder:
    torch.manual_seed(468)

    return Decoder(
        vocab_size=50,
        embedding_dim=8,
        encoder_dim=24,
        decoder_hidden_dim=16,
        attention_dim=20,
        pad_id=0,
        dropout=0.0,
    )


def create_inputs():
    input_ids = torch.tensor([2, 4])
    hidden = torch.randn(2, 16)
    cell = torch.randn(2, 16)
    encoder_outputs = torch.randn(2, 5, 24)
    source_mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
    )

    return (
        input_ids,
        hidden,
        cell,
        encoder_outputs,
        source_mask,
    )


def test_decoder_output_shapes() -> None:
    decoder = create_decoder()
    inputs = create_inputs()

    logits, hidden, cell, weights = decoder(*inputs)

    assert logits.shape == (2, 50)
    assert hidden.shape == (2, 16)
    assert cell.shape == (2, 16)
    assert weights.shape == (2, 5)


def test_decoder_attention_respects_mask() -> None:
    decoder = create_decoder()
    inputs = create_inputs()

    _, _, _, weights = decoder(*inputs)

    assert torch.allclose(
        weights.sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )
    assert torch.allclose(
        weights[1, 3:],
        torch.zeros(2),
        atol=1e-7,
    )


def test_decoder_supports_backpropagation() -> None:
    decoder = create_decoder()
    inputs = create_inputs()

    logits, _, _, _ = decoder(*inputs)
    loss = logits.square().mean()
    loss.backward()

    assert decoder.embedding.weight.grad is not None
    assert (
        decoder.attention.score_projection.weight.grad
        is not None
    )