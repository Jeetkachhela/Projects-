from src.text_extraction import find_target_in_text


def test_full_with_suffix():
    text = "ID: 160390797970200578_1_gsm something"
    res = find_target_in_text(text, source="test", base_confidence=0.9)
    assert res is not None
    assert res.matched_text == "160390797970200578_1_gsm"


def test_full_without_suffix():
    text = "Order: 161820476409495744_1 shipped"
    res = find_target_in_text(text, source="test", base_confidence=0.9)
    assert res is not None
    assert res.matched_text == "161820476409495744_1"


def test_reject_short_number():
    text = "Bad value 123456_1_gsm"
    res = find_target_in_text(text, source="test", base_confidence=0.9)
    assert res is None
