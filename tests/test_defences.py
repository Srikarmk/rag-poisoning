from ragpoison.defences import agreement_answer, is_instruction, strip_instructions
from ragpoison.poisons import BROAD_GENERIC_TEXT, BROAD_INSTRUCTION, INJECTION_TEMPLATE

WIKI = ("Super Bowl 50 was an American football game to determine the champion of the National Football "
        "League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos "
        "defeated the National Football Conference (NFC) champion Carolina Panthers 24-10.")


def test_filter_removes_injection_and_keeps_question():
    text = INJECTION_TEMPLATE.format(q="Who won Super Bowl 50?", t="Super Bowl 50", canary="QX-1234")
    out = strip_instructions(text)
    assert "QX-1234" not in out and "ignored" not in out
    assert "Who won Super Bowl 50?" in out


def test_filter_removes_broad_instruction():
    text = BROAD_GENERIC_TEXT + BROAD_INSTRUCTION.format(canary="BRD-0410")
    out = strip_instructions(text)
    assert "BRD-0410" not in out and "General reference guide" in out


def test_filter_leaves_ordinary_text_alone():
    assert strip_instructions(WIKI) == WIKI
    assert not is_instruction("Answers to the census were collected in 1901.")
    assert not is_instruction("The council voted to ignore the 1890 ruling.")


def test_agreement_needs_two_passages():
    ps = ["The Denver Broncos won.", "Denver Broncos champions.", "Velkar won."]
    assert agreement_answer("Denver Broncos", ps) == "Denver Broncos"
    assert agreement_answer("Velkar", ps) is None
