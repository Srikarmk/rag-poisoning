import numpy as np
import pandas as pd

from ragpoison import poisons


def _corpus():
    return pd.DataFrame({"pid": ["p0", "p1"], "title": ["Moon", "Sun"],
                         "text": ["The Moon landing happened in 1969 with three astronauts.",
                                  "The Sun is a star at the centre of the Solar System."],
                         "origin": ["validation", "validation"]})


def _targets():
    return pd.DataFrame({"qid": ["q1"], "question": ["When did the Moon landing happen?"],
                         "answers": [["1969"]], "title": ["Moon"], "gold_pid": ["p0"]})


def test_one_in_n_inserts_exactly_one_document():
    corpus = _corpus()
    kinds = poisons.build_targeted(_targets(), corpus, seed=1)["q1"]
    assert len(kinds["false_fact"]) == 1 and len(kinds["injection"]) == 1
    assert len(kinds["false_fact_x5"]) == 5
    planted = poisons.insert(corpus, kinds["false_fact"])
    assert len(planted) == len(corpus) + 1
    assert (planted["origin"] == "planted").sum() == 1
    pd.testing.assert_frame_equal(planted.iloc[:len(corpus)], corpus)


def test_insert_rejects_pid_collision():
    corpus = _corpus()
    doc = poisons.Poison("p0", "false_fact", "x", "Moon")
    try:
        poisons.insert(corpus, [doc])
    except ValueError:
        return
    raise AssertionError("collision not caught")


def test_fake_year_differs_and_is_not_in_passage():
    rng = np.random.default_rng(0)
    fake, keys = poisons.make_fake(["1969"], "The Moon landing happened in 1969.", rng, "")
    assert fake != "1969" and keys == [fake] and fake not in "the moon landing happened in 1969."


def test_fake_number_and_name():
    rng = np.random.default_rng(0)
    fake, keys = poisons.make_fake(["three"], "three astronauts", rng, "")
    assert keys[0].isdigit() and keys[0] != "3"
    fake, keys = poisons.make_fake(["Neil Armstrong"], "Neil Armstrong walked", rng, "neil armstrong walked")
    assert len(fake.split()) == 2 and fake[0].isupper()
    assert keys[0] not in "neil armstrong walked"


def test_false_fact_contains_question_and_fake():
    kinds = poisons.build_targeted(_targets(), _corpus(), seed=1)["q1"]
    ff = kinds["false_fact"][0]
    assert "When did the Moon landing happen?" in ff.text and ff.fake_answer in ff.text
    inj = kinds["injection"][0]
    assert inj.keys[0].upper() in inj.text


def test_broad_poisons_are_single_documents():
    broad = poisons.build_broad(_corpus())
    assert [d.kind for d in broad] == ["broad_generic", "broad_terms"]
    assert all(d.keys[0].upper() in d.text for d in broad)
