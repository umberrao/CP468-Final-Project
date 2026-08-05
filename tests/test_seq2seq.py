import torch

from src.models.seq2seq import Seq2Seq


def create_model() -> Seq2Seq:
    torch.manual_seed(468)

    return Seq2Seq(
        vocab_size=50,
        embedding_dim=8,
        encoder_hidden_dim=12,
        decoder_hidden_dim=16,
        attention_dim=20,
        pad_id=0,
        dropout=0.0,
    )


def create_inputs():
    source_ids = torch.tensor(
        [
            [2, 4, 5, 6, 3],
            [2, 7, 3, 0, 0],
        ]
    )
    source_lengths = torch.tensor([5, 3])
    source_mask = source_ids.ne(0)
    decoder_input_ids = torch.tensor(
        [
            [2, 8, 9, 10],
            [2, 11, 12, 13],
        ]
    )

    return (
        source_ids,
        source_lengths,
        source_mask,
        decoder_input_ids,
    )


def test_seq2seq_output_shapes() -> None:
    model = create_model()
    inputs = create_inputs()

    logits, attention = model(*inputs)

    assert logits.shape == (2, 4, 50)
    assert attention.shape == (2, 4, 5)


def test_seq2seq_attention_ignores_padding() -> None:
    model = create_model()
    inputs = create_inputs()

    _, attention = model(*inputs)

    assert torch.allclose(
        attention.sum(dim=2),
        torch.ones(2, 4),
        atol=1e-6,
    )
    assert torch.allclose(
        attention[1, :, 3:],
        torch.zeros(4, 2),
        atol=1e-7,
    )


def test_seq2seq_supports_backpropagation() -> None:
    model = create_model()
    inputs = create_inputs()

    logits, _ = model(*inputs)
    loss = logits.square().mean()
    loss.backward()

    assert model.encoder.embedding.weight.grad is not None
    assert model.decoder.embedding.weight.grad is not None


def test_greedy_decoding_stops_at_eos() -> None:
    model = create_model()
    model.eval()

    with torch.no_grad():
        model.decoder.output_projection.weight.zero_()
        model.decoder.output_projection.bias.zero_()
        model.decoder.output_projection.bias[3] = 10.0

    source_ids, source_lengths, source_mask, _ = (
        create_inputs()
    )

    generated, attention = model.greedy_decode(
        source_ids,
        source_lengths,
        source_mask,
        bos_id=2,
        eos_id=3,
        max_length=5,
    )

    assert generated.shape == (2, 1)
    assert attention.shape == (2, 1, 5)
    assert torch.all(generated == 3)