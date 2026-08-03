from src.text import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    Vocabulary,
    detokenize,
    tokenize,
)


def test_tokenize_and_detokenize() -> None:
    text = "A well-known cat's toy."
    tokens = tokenize(text)

    assert tokens == [
        "a",
        "well-known",
        "cat's",
        "toy",
        ".",
    ]
    assert detokenize(tokens) == "a well-known cat's toy."


def test_vocabulary_special_tokens() -> None:
    vocabulary = Vocabulary.build(
        ["Cats chase mice.", "Dogs chase cats."],
        min_frequency=1,
    )

    assert vocabulary.id_to_token[:4] == [
        PAD_TOKEN,
        UNK_TOKEN,
        BOS_TOKEN,
        EOS_TOKEN,
    ]
    assert vocabulary.pad_id == 0
    assert vocabulary.unk_id == 1


def test_encode_unknown_and_max_length() -> None:
    vocabulary = Vocabulary.build(
        ["known words"],
        min_frequency=1,
    )

    encoded = vocabulary.encode(
        "known mystery",
        max_length=4,
    )

    assert encoded[0] == vocabulary.bos_id
    assert encoded[2] == vocabulary.unk_id
    assert encoded[-1] == vocabulary.eos_id
    assert len(encoded) == 4


def test_vocabulary_save_and_load(tmp_path) -> None:
    original = Vocabulary.build(
        ["simple sentence", "another sentence"],
        min_frequency=1,
    )

    path = tmp_path / "vocabulary.json"
    original.save(path)
    loaded = Vocabulary.load(path)

    assert loaded.id_to_token == original.id_to_token
    assert loaded.token_to_id == original.token_to_id