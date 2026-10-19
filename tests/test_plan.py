"""Checks on the real experiment plan; skipped until `poison fetch` has been run."""

import pytest

from ragpoison import config as C

pytestmark = pytest.mark.skipif(not (C.RAW / "squad_validation.parquet").exists(), reason="data not fetched")


@pytest.fixture(scope="module")
def study():
    from ragpoison.study import Study
    return Study()


def test_corpus_is_ten_thousand_and_holds_every_gold_passage(study):
    assert len(study.corpus) == C.CORPUS_SIZE
    assert set(study.selected["gold_pid"]) <= set(study.corpus["pid"])


def test_single_poison_jobs_plant_exactly_one_document(study):
    from ragpoison.experiment import plan_main
    jobs = plan_main(study)
    for j in jobs:
        if j.cond in ("false_fact", "injection", "broad_generic", "broad_terms"):
            assert len(j.extras) == 1
        elif j.cond == "false_fact_x5":
            assert len(j.extras) == C.DOSE_COPIES
        elif j.cond == "clean":
            assert j.extras == []


def test_retrieval_with_one_poison_changes_at_most_one_slot(study):
    t = study.targets.iloc[0]
    doc = study.targeted[t["qid"]]["false_fact"][0]
    clean = [p.pid for p in study.retrieve(t["qid"], [], 5)]
    poisoned = [p.pid for p in study.retrieve(t["qid"], [doc], 5)]
    assert doc.pid in poisoned
    assert [p for p in poisoned if p != doc.pid] == clean[:4]


def test_made_up_answers_never_appear_in_the_corpus(study):
    text = " ".join(study.corpus["text"]).lower()
    for kinds in study.targeted.values():
        ff = kinds["false_fact"][0]
        if not ff.keys[0].isdigit():
            assert ff.keys[0] not in text


def test_reduced_profile_is_a_subset(study):
    from ragpoison.experiment import plan_main
    full = {(j.cond, j.qid, j.k) for j in plan_main(study)}
    red = {(j.cond, j.qid, j.k) for j in plan_main(study, reduced=True)}
    assert red <= full and all(k == C.MAIN_K for _, _, k in red)
