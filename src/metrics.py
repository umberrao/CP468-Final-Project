from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from sacrebleu.metrics import BLEU


Ngram = tuple[str, ...]

_SARI_TOKENIZER = BLEU(tokenize="13a").tokenizer


def _normalize(text: str) -> list[str]:
    """Lowercase and tokenize text consistently for SARI."""
    return _SARI_TOKENIZER(text.lower().strip()).split()


def _ngrams(
    tokens: Sequence[str],
    order: int,
) -> list[Ngram]:
    return [
        tuple(tokens[index:index + order])
        for index in range(len(tokens) - order + 1)
    ]


def _f1(precision: float, recall: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0

    return (
        2.0
        * precision
        * recall
        / (precision + recall)
    )


def _sari_ngram_score(
    source: Sequence[Ngram],
    prediction: Sequence[Ngram],
    references: Sequence[Sequence[Ngram]],
) -> tuple[float, float, float]:
    reference_count = len(references)

    reference_grams = Counter(
        gram
        for reference in references
        for gram in reference
    )

    source_counts = Counter(source)
    prediction_counts = Counter(prediction)

    source_repeated = Counter(
        {
            gram: count * reference_count
            for gram, count in source_counts.items()
        }
    )
    prediction_repeated = Counter(
        {
            gram: count * reference_count
            for gram, count in prediction_counts.items()
        }
    )

    # KEEP
    kept = source_repeated & prediction_repeated
    kept_correct = kept & reference_grams
    kept_reference = source_repeated & reference_grams

    keep_precision = 1.0

    if kept:
        keep_precision = sum(
            kept_correct[gram] / kept[gram]
            for gram in kept_correct
        ) / len(kept)

    keep_recall = 1.0

    if kept_reference:
        keep_recall = (
            sum(kept_correct.values())
            / sum(kept_reference.values())
        )

    keep_f1 = _f1(
        keep_precision,
        keep_recall,
    )

    # DELETE
    deleted = source_repeated - prediction_repeated
    deleted_correct = deleted - reference_grams

    delete_precision = 1.0

    if deleted:
        delete_precision = sum(
            deleted_correct[gram] / deleted[gram]
            for gram in deleted_correct
        ) / len(deleted)

    # ADD
    added = (
        set(prediction_counts)
        - set(source_counts)
    )
    added_correct = added & set(reference_grams)
    added_reference = (
        set(reference_grams)
        - set(source_counts)
    )

    add_precision = (
        1.0
        if not added
        else len(added_correct) / len(added)
    )
    add_recall = (
        1.0
        if not added_reference
        else len(added_correct) / len(added_reference)
    )

    add_f1 = _f1(
        add_precision,
        add_recall,
    )

    return keep_f1, delete_precision, add_f1


def sari_sentence(
    source: str,
    prediction: str,
    references: Sequence[str],
) -> float:
    """Compute sentence-level SARI from 0 to 100."""
    if not references:
        raise ValueError(
            "SARI requires at least one reference."
        )

    source_tokens = _normalize(source)
    prediction_tokens = _normalize(prediction)
    reference_tokens = [
        _normalize(reference)
        for reference in references
    ]

    keep_scores = []
    delete_scores = []
    add_scores = []

    for order in range(1, 5):
        keep, delete, add = _sari_ngram_score(
            _ngrams(source_tokens, order),
            _ngrams(prediction_tokens, order),
            [
                _ngrams(tokens, order)
                for tokens in reference_tokens
            ],
        )

        keep_scores.append(keep)
        delete_scores.append(delete)
        add_scores.append(add)

    sari = (
        sum(keep_scores) / 4
        + sum(delete_scores) / 4
        + sum(add_scores) / 4
    ) / 3

    return 100.0 * sari


def corpus_sari(
    sources: Sequence[str],
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> float:
    """Compute mean SARI over a dataset."""
    if not (
        len(sources)
        == len(predictions)
        == len(references)
    ):
        raise ValueError(
            "Sources, predictions, and references "
            "must have equal lengths."
        )

    if not predictions:
        raise ValueError(
            "Cannot evaluate an empty dataset."
        )

    return sum(
        sari_sentence(source, prediction, refs)
        for source, prediction, refs in zip(
            sources,
            predictions,
            references,
        )
    ) / len(predictions)


def corpus_bleu(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> float:
    """Compute lowercase SacreBLEU with multiple references."""
    if len(predictions) != len(references):
        raise ValueError(
            "Predictions and references "
            "must have equal lengths."
        )

    if not predictions:
        raise ValueError(
            "Cannot evaluate an empty dataset."
        )

    reference_count = len(references[0])

    if reference_count == 0:
        raise ValueError(
            "BLEU requires at least one reference."
        )

    if any(
        len(row) != reference_count
        for row in references
    ):
        raise ValueError(
            "Each example must have the same "
            "number of references."
        )

    reference_streams = [
        [
            example_references[index]
            for example_references in references
        ]
        for index in range(reference_count)
    ]

    metric = BLEU(
        lowercase=True,
        tokenize="13a",
    )

    return float(
        metric.corpus_score(
            list(predictions),
            reference_streams,
        ).score
    )