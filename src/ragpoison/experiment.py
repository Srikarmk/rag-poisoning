"""Plans and runs every condition for one model, writing one row per model call."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from .checks import adopted, is_correct, support_count
from .defences import strip_instructions
from .llm import NotCached, Ollama
from .poisons import Poison
from .prompts import build_messages, parse_answer
from .retrieve import reach_mask
from .study import Passage, Study


@dataclass
class Job:
    phase: str
    cond: str
    qid: str
    k: int
    owner: str = ""
    extras: list[Poison] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)
    style: str = "base"
    filt: bool = False
    rep: int = 0
    forced: bool = False


def model_slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def _owner_keys(study: Study, owner: str, cond: str) -> list[str]:
    if not owner:
        return []
    kinds = study.targeted[owner]
    return kinds["injection"][0].keys if cond == "injection" else kinds["false_fact"][0].keys


def broad_subset(study: Study, n: int) -> set[str]:
    """A fixed, seeded subset of the broad questions; smaller subsets nest inside larger ones."""
    qids = sorted(study.selected.loc[study.selected["set"] == "broad", "qid"])
    order = np.random.default_rng(C.SEED + 3).permutation(len(qids))
    return {qids[i] for i in order[:n]}


def selected(study: Study, reduced: bool):
    sel = study.selected
    if reduced:
        keep = broad_subset(study, C.REDUCED_BROAD)
        sel = sel[(sel["set"] != "broad") | sel["qid"].isin(keep)]
    return sel


def plan_main(study: Study, reduced: bool = False) -> list[Job]:
    jobs: list[Job] = []
    sel = selected(study, reduced)
    ks = (C.MAIN_K,) if reduced else C.KS
    for r in sel.itertuples():
        jobs.append(Job("main", "clean", r.qid, C.MAIN_K, r.target_qid, [], _owner_keys(study, r.target_qid, "clean")))
    for r in study.targets.itertuples():
        for k in ks:
            if k != C.MAIN_K:
                jobs.append(Job("main", "clean", r.qid, k, r.qid, [], _owner_keys(study, r.qid, "clean")))
    for t in study.targets.itertuples():
        kinds = study.targeted[t.qid]
        near = sel[(sel["target_qid"] == t.qid) & (sel["set"] != "target")]
        for cond in ("false_fact", "injection"):
            for k in ks:
                jobs.append(Job("main", cond, t.qid, k, t.qid, kinds[cond], kinds[cond][0].keys))
            for n in near.itertuples():
                jobs.append(Job("main", cond, n.qid, C.MAIN_K, t.qid, kinds[cond], kinds[cond][0].keys))
        x5 = kinds["false_fact_x5"]
        jobs.append(Job("main", "false_fact_x5", t.qid, C.MAIN_K, t.qid, x5, x5[0].keys))
        for n in near.itertuples():
            jobs.append(Job("main", "false_fact_x5", n.qid, C.MAIN_K, t.qid, x5, x5[0].keys))
    # Unrelated questions that a targeted poison happens to reach.
    broad = sel[sel["set"] == "broad"]
    bvecs = np.stack([study.qvec(q) for q in broad["qid"]])
    bscores = np.stack([study.clean_scores[study.row_of[q]] for q in broad["qid"]])
    for t in study.targets.itertuples():
        for cond in ("false_fact", "injection"):
            doc = study.targeted[t.qid][cond][0]
            hit = reach_mask(bvecs, bscores, study.extra_vec(doc.pid), C.MAIN_K)
            for q in broad["qid"].values[hit]:
                jobs.append(Job("main", cond, q, C.MAIN_K, t.qid, [doc], doc.keys))
    for doc in study.broad:
        for r in sel.itertuples():
            jobs.append(Job("main", doc.kind, r.qid, C.MAIN_K, "", [doc], doc.keys))
    # Diagnostic: the broad poison forced into rank 1, to separate retrieval from model behaviour.
    forced = broad_subset(study, C.REDUCED_FORCED if reduced else C.FORCED_BROAD)
    for doc in study.broad:
        for q in sorted(forced):
            jobs.append(Job("main", doc.kind + "_forced", q, C.MAIN_K, "", [doc], doc.keys, forced=True))
    return jobs


def plan_defences(study: Study, main_jobs: list[Job], reduced: bool = False) -> list[Job]:
    """Defences on the clean questions, the targeted poisons on their own target, and the forced broad poison.

    The instruction filter runs on every clean question; the quote prompt on targets, nearby and a
    subset of the broad set (reduced profile: targets and nearby only). Forced broad poisons get the
    filter only.
    """
    quote_broad = broad_subset(study, C.QUOTE_BROAD)
    qset = dict(zip(study.selected["qid"], study.selected["set"]))
    out = []
    for j in main_jobs:
        if j.k != C.MAIN_K:
            continue
        if j.cond == "clean":
            out.append(Job("defence", j.cond, j.qid, j.k, j.owner, j.extras, j.keys, "base", True))
            if qset[j.qid] != "broad" or (j.qid in quote_broad and not reduced):
                out.append(Job("defence", j.cond, j.qid, j.k, j.owner, j.extras, j.keys, "quote", False))
        elif j.cond in ("false_fact", "injection", "false_fact_x5") and j.qid == j.owner:
            out.append(Job("defence", j.cond, j.qid, j.k, j.owner, j.extras, j.keys, "base", True))
            out.append(Job("defence", j.cond, j.qid, j.k, j.owner, j.extras, j.keys, "quote", False))
        elif j.forced:
            out.append(Job("defence", j.cond, j.qid, j.k, j.owner, j.extras, j.keys, "base", True, forced=True))
    return out


def plan_repeat(study: Study, main_jobs: list[Job], reduced: bool = False) -> list[Job]:
    """Second run of the clean condition and of the targeted poisons on their targets.

    Full profile: clean on targets, nearby and part of the broad set. Reduced profile: clean on targets only.
    """
    rep_broad = broad_subset(study, C.REPEAT_BROAD)
    qset = dict(zip(study.selected["qid"], study.selected["set"]))
    out = []
    for j in main_jobs:
        if j.k != C.MAIN_K or j.forced:
            continue
        if reduced:
            clean = j.cond == "clean" and qset[j.qid] == "target"
        else:
            clean = j.cond == "clean" and (qset[j.qid] != "broad" or j.qid in rep_broad)
        poisoned = j.cond in ("false_fact", "injection") and j.qid == j.owner
        if clean or poisoned:
            out.append(Job("repeat", j.cond, j.qid, j.k, j.owner, j.extras, j.keys, j.style, j.filt, 1))
    return out


def plan_hop2(study: Study, rows: list[dict]) -> list[Job]:
    """Write each adopted false-fact answer back as a note (poison removed) and ask again."""
    sel = study.selected
    jobs = []
    for r in rows:
        if not (r["phase"] == "main" and r["cond"] == "false_fact" and r["k"] == C.MAIN_K
                and r["qid"] == r["owner"] and r["adopted"]):
            continue
        t = study.targets.set_index("qid").loc[r["qid"]]
        keys = study.targeted[r["qid"]]["false_fact"][0].keys
        text = (f"Notes from an earlier answer about {t['title']}. Question: {t['question']} "
                f"Answer: {r['answer']}")
        # The pid carries a hash of the text: notes differ between models and the Study caches by pid.
        pid = f"x-note-{r['qid']}-{hashlib.sha256(text.encode()).hexdigest()[:10]}"
        note = Poison(pid, "note", text, "Notes", r["qid"], keys)
        for q in [r["qid"]] + sel[(sel["target_qid"] == r["qid"]) & (sel["set"] != "target")]["qid"].tolist():
            jobs.append(Job("hop2", "note", q, C.MAIN_K, r["qid"], [note], keys))
    return jobs


def run_job(study: Study, client: Ollama, model: str, j: Job, cache_only: bool = False) -> dict:
    if j.forced:
        doc = j.extras[0]
        study.register([doc])
        clean = study.retrieve(j.qid, [], j.k - 1)
        passages = [Passage(doc.pid, doc.title, doc.text, float("nan"), True)] + clean
    else:
        passages = study.retrieve(j.qid, j.extras, j.k)
    shown = [(p.title, strip_instructions(p.text) if j.filt else p.text) for p in passages]
    question = study.questions.iloc[study.row_of[j.qid]]
    res = client.chat(model, build_messages(question["question"], shown, j.style), rep=j.rep,
                      cache_only=cache_only)
    answer = parse_answer(res["content"])
    planted_ranks = [i + 1 for i, p in enumerate(passages) if p.planted]
    return {
        "model": model, "phase": j.phase, "cond": j.cond, "defence": "quote" if j.style == "quote"
        else ("filter" if j.filt else "none"), "rep": j.rep, "k": j.k, "qid": j.qid,
        "qset": question["set"], "owner": j.owner, "in_topk": bool(planted_ranks),
        "n_planted": len(planted_ranks), "planted_rank": planted_ranks[0] if planted_ranks else 0,
        "retrieved": [p.pid for p in passages], "output": res["content"], "answer": answer,
        "correct": is_correct(answer, list(question["answers"])),
        "adopted": adopted(answer, j.keys) if j.keys else False,
        "obeyed": adopted(res["content"], j.keys) if j.keys else False,
        "support": support_count(answer, [t for _, t in shown]),
        "filtered_changed": j.filt and any(a != b.text for (_, a), b in zip(shown, passages)),
        "forced": j.forced, "cached": res["cached"], "call": res["key"][:16],
        "wall": res["meta"].get("wall"),
    }


def run(study: Study, model: str, phases=("main", "hop2", "defence", "repeat"), client: Ollama | None = None,
        log=sys.stderr, reduced: bool = False, cache_only: bool = False) -> list[dict]:
    """Runs the phases in order. With cache_only, a phase with any uncached call is left out whole."""
    client = client or Ollama(cache_path=C.CACHE / f"llm_{model_slug(model)}.sqlite")
    rows: list[dict] = []
    main = plan_main(study, reduced)
    t0 = time.time()
    done, missing = [], {}

    def execute(jobs, label):
        n0 = client.misses
        out, lacking = [], 0
        for i, j in enumerate(jobs, 1):
            try:
                out.append(run_job(study, client, model, j, cache_only))
            except NotCached:
                lacking += 1
            if i % 200 == 0 or i == len(jobs):
                print(f"[{model}] {label} {i}/{len(jobs)} new calls={client.misses - n0} "
                      f"elapsed={time.time() - t0:.0f}s", file=log, flush=True)
        if lacking:
            missing[label] = lacking
        else:
            rows.extend(out)
            done.append(label)

    if "main" in phases:
        execute(main, "main")
    if "hop2" in phases and "main" in done:
        execute(plan_hop2(study, rows), "hop2")
    if "defence" in phases:
        execute(plan_defences(study, main, reduced), "defence")
    if "repeat" in phases:
        execute(plan_repeat(study, main, reduced), "repeat")
    out = C.RESULTS / model_slug(model)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"model": model, "profile": "reduced" if reduced else "full", "phases_done": done,
            "phases_missing_calls": missing, "digest": client.model_digest(model), "options": C.LLM_OPTIONS,
            "think": False if client._thinks(model, cache_only) else None,
            "rows": len(rows), "new_calls": client.misses, "cache_hits": client.hits,
            "elapsed_s": round(time.time() - t0, 1),
            "distinct_calls": len({r["call"] for r in rows}),
            "model_seconds_all_calls": round(sum({r["call"]: r["wall"] or 0 for r in rows}.values()), 1)}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return rows
