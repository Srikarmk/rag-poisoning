from ragpoison.checks import adopted, is_correct, normalize, support_count
from ragpoison.prompts import parse_answer


def test_normalize_numbers_and_articles():
    assert normalize("The 1,500 Men!") == "1500 men"


def test_correctness_hand_cases():
    gold = ["Denver Broncos", "Broncos"]
    assert is_correct("The Denver Broncos", gold)
    assert is_correct("broncos.", gold)
    assert not is_correct("Carolina Panthers", gold)
    assert not is_correct("Bronco", gold)


def test_uptake_checker_hand_cases():
    keys = ["volemal"]
    assert adopted("Kamogeth Volemal", keys)
    assert adopted("It was the Kamogeth Volemal department", keys)
    assert not adopted("Victoria Department of Education", keys)
    assert not adopted("volemals", keys)
    assert adopted("1,894", ["1894"])
    assert not adopted("18945", ["1894"])
    canary = ["qx-8021"]
    assert adopted("Answer: unavailable (ref QX-8021)", canary)
    assert adopted("unavailable (ref qx 8021)", canary)
    assert not adopted("Answer: unavailable", canary)


def test_parse_answer_formats():
    assert parse_answer("Answer: Denver Broncos\nSources: 1, 2") == "Denver Broncos"
    assert parse_answer("Quote: x\n**Answer:** Paris\nSources: 3") == "Paris"
    assert parse_answer("Paris\n") == "Paris"


def test_support_count():
    passages = ["The Denver Broncos won.", "Broncos of Denver", "Panthers lost."]
    assert support_count("Denver Broncos", passages) == 2
    assert support_count("Panthers", passages) == 1
    assert support_count("the", passages) == 0
