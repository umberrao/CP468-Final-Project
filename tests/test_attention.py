import torch

from src.models.attention import AdditiveAttention


def create_attention() -> AdditiveAttention:
    torch.manual_seed(468)

    return AdditiveAttention(
        encoder_dim=24,
        decoder_dim=16,
        attention_dim=20,
    )


def test_attention_output_shapes() -> None:
    attention = create_attention()

    encoder_outputs = torch.randn(2, 5, 24)
    decoder_hidden = torch.randn(2, 16)
    source_mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
    )

    context, weights = attention(
        decoder_hidden,
        encoder_outputs,
        source_mask,
    )

    assert context.shape == (2, 24)
    assert weights.shape == (2, 5)


def test_attention_weights_sum_to_one() -> None:
    attention = create_attention()

    encoder_outputs = torch.randn(2, 5, 24)
    decoder_hidden = torch.randn(2, 16)
    source_mask = torch.ones(2, 5, dtype=torch.bool)

    _, weights = attention(
        decoder_hidden,
        encoder_outputs,
        source_mask,
    )

    assert torch.allclose(
        weights.sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )


def test_attention_ignores_padding() -> None:
    attention = create_attention()

    encoder_outputs = torch.randn(2, 5, 24)
    decoder_hidden = torch.randn(2, 16)
    source_mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
    )

    _, weights = attention(
        decoder_hidden,
        encoder_outputs,
        source_mask,
    )

    assert torch.allclose(
        weights[1, 3:],
        torch.zeros(2),
        atol=1e-7,
    )