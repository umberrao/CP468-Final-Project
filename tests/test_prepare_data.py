from scripts.prepare_data import normalize_row


def test_normalize_row_preserves_duplicate_references() -> None:
    row = normalize_row(
        {
            "gem_id": "test-1",
            "source": "A complex sentence.",
            "target": "A simple sentence.",
            "references": [
                "A simple sentence.",
                "A simple sentence.",
                "Another simple sentence.",
            ],
        }
    )

    assert row is not None
    assert row["references"] == [
        "A simple sentence.",
        "A simple sentence.",
        "Another simple sentence.",
    ]