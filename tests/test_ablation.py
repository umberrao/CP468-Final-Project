import torch

from src.models.seq2seq import Seq2Seq


def build_model(use_attention: bool) -> Seq2Seq:
    return Seq2Seq(
        vocab_size=32,
        embedding_dim=8,
        encoder_hidden_dim=8,
        decoder_hidden_dim=8,
        attention_dim=8,
        pad_id=0,
        dropout=0.0,
        use_attention=use_attention,
    )


def test_no_attention_forward() -> None:
    model = build_model(False)

    source_ids = torch.tensor(
        [
            [2, 5, 6, 3],
            [2, 7, 3, 0],
        ]
    )
    source_lengths = torch.tensor([4, 3])
    source_mask = source_ids.ne(0)

    decoder_inputs = torch.tensor(
        [
            [2, 8, 9],
            [2, 10, 3],
        ]
    )

    logits, attention = model(
        source_ids,
        source_lengths,
        source_mask,
        decoder_inputs,
    )

    assert logits.shape == (2, 3, 32)
    assert attention.shape == (2, 3, 4)
    assert torch.count_nonzero(attention) == 0
    assert model.decoder.attention is None


def test_ablation_has_fewer_parameters() -> None:
    attention_model = build_model(True)
    ablation_model = build_model(False)

    attention_parameters = sum(
        parameter.numel()
        for parameter in attention_model.parameters()
    )
    ablation_parameters = sum(
        parameter.numel()
        for parameter in ablation_model.parameters()
    )

    assert ablation_parameters < attention_parameters