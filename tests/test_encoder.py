import torch

from src.models.encoder import Encoder


def create_encoder() -> Encoder:
    torch.manual_seed(468)

    return Encoder(
        vocab_size=50,
        embedding_dim=8,
        encoder_hidden_dim=12,
        decoder_hidden_dim=16,
        pad_id=0,
        dropout=0.0,
    )


def test_encoder_output_shapes() -> None:
    encoder = create_encoder()

    source_ids = torch.tensor(
        [
            [2, 4, 5, 6, 3],
            [2, 7, 3, 0, 0],
        ]
    )
    source_lengths = torch.tensor([5, 3])

    outputs, hidden, cell = encoder(
        source_ids,
        source_lengths,
    )

    assert outputs.shape == (2, 5, 24)
    assert hidden.shape == (2, 16)
    assert cell.shape == (2, 16)


def test_encoder_padding_outputs_are_zero() -> None:
    encoder = create_encoder()

    source_ids = torch.tensor(
        [
            [2, 4, 5, 6, 3],
            [2, 7, 3, 0, 0],
        ]
    )
    source_lengths = torch.tensor([5, 3])

    outputs, _, _ = encoder(
        source_ids,
        source_lengths,
    )

    assert torch.all(outputs[1, 3:] == 0)