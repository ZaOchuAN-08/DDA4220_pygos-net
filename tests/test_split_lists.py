from pathlib import Path


def test_reconstructed_split_lists_are_complete_and_disjoint():
    root = Path(__file__).resolve().parents[1] / "data" / "splits"
    train = (root / "train.txt").read_text().splitlines()
    val = (root / "val.txt").read_text().splitlines()
    test = (root / "test.txt").read_text().splitlines()
    assert (len(train), len(val), len(test)) == (446, 146, 146)
    assert len(set(train + val + test)) == 738
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
