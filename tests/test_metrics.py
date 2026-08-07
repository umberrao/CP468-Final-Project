import pytest

from src.metrics import (
    corpus_bleu,
    corpus_sari,
    sari_sentence,
)


def test_sari_perfect_match() -> None:
    text = "The cat is sitting on the mat."

    score = sari_sentence(
        text,
        text,
        [text],
    )

    assert score == pytest.approx(100.0)


def test_sari_known_example() -> None:
    source = "About 95 species are currently accepted ."
    prediction = "About 95 you now get in ."
    references = [
        "About 95 species are currently known .",
        "About 95 species are now accepted .",
        "95 species are now accepted .",
    ]

    score = sari_sentence(
        source,
        prediction,
        references,
    )

    assert score == pytest.approx(
        26.953601953601954
    )


def test_bleu_perfect_match() -> None:
    predictions = [
        "this is a complete simple sentence .",
        "another sufficiently long simple sentence .",
    ]

    references = [
        [predictions[0], predictions[0]],
        [predictions[1], predictions[1]],
    ]

    assert corpus_bleu(
        predictions,
        references,
    ) == pytest.approx(100.0)


def test_corpus_sari_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        corpus_sari(
            ["source"],
            [],
            [["reference"]],
        )