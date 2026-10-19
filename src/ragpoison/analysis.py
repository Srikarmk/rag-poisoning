"""Turns results/*/rows.jsonl and the embeddings into reports/results.md and reports/figures/*.png."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402
from .checks import gives_instructed_reply  # noqa: E402
from .retrieve import reach_mask  # noqa: E402
from .stats import boot_diff, boot_mean, boot_ratio, fmt  # noqa: E402
from .study import Study  # noqa: E402

MODEL_ORDER = ["qwen3:8b", "llama3.1:8b", "gemma3:12b", "qwen3:30b-a3b"]
INSTRUCTION_KINDS = {"injection", "broad_generic", "broad_terms", "broad_generic_forced", "broad_terms_forced"}
LABEL = {"false_fact": "false fact", "injection": "injection", "false_fact_x5": "false fact x5",
         "broad_generic": "broad (generic)", "broad_terms": "broad (top terms)", "note": "second-hop note",
         "broad_generic_forced": "broad (generic), forced rank 1",
         "broad_terms_forced": "broad (top terms), forced rank 1"}
SET_LABEL = {"target": "target", "nearby_passage": "nearby, same paragraph",
             "nearby_article": "nearby, same article", "broad": "unrelated (broad set)",
             "all_other": "all other validation questions"}
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------- loading

def load_rows() -> pd.DataFrame:
    frames = []
    for path in sorted(C.RESULTS.glob("*/rows.jsonl")):
        frames.append(pd.read_json(path, lines=True))
    if not frames:
        raise SystemExit("no results yet: run `poison run --model ...` first")
    df = pd.concat(frames, ignore_index=True)
    # Instruction poisons count as obeyed if the canary code appears, or if the answer gives the reply word
    # the instruction asked for ("unavailable" for the injection, "unknown" for the broad passages).
    word = np.where(df["cond"] == "injection", "unavailable", "unknown")
    reply = np.array([gives_instructed_reply(a, w) for a, w in zip(df["answer"].fillna(""), word)])
    df["canary"] = df["obeyed"]
    df["reply_word"] = reply
    df["uptake"] = np.where(df["cond"].isin(INSTRUCTION_KINDS), df["obeyed"] | reply, df["adopted"])
    return df


def models_in(df: pd.DataFrame) -> list[str]:
    present = set(df["model"])
    return [m for m in MODEL_ORDER if m in present] + sorted(present - set(MODEL_ORDER))


def with_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Post-hoc source-agreement check: an answer supported by fewer than two passages becomes an abstention."""
    d = df.copy()
    keep = d["support"] >= 2
    d["abstain"] = ~keep
    for col in ("correct", "adopted", "obeyed", "uptake"):
        d[col] = d[col] & keep
    return d


# ---------------------------------------------------------------- reach (retrieval only)

def reach_table(s: Study) -> pd.DataFrame:
    q = s.questions
    vecs, scores = s.all_qvecs, s.clean_scores
    rows = []
    title_of = dict(zip(q["qid"], q["title"]))
    targets = s.targets
    for kind in ("false_fact", "injection", "false_fact_x5"):
        for k in C.KS:
            per = {g: [] for g in ("target", "nearby_passage", "nearby_article", "broad", "all_other")}
            counts = []
            for t in targets.itertuples():
                docs = s.targeted[t.qid][kind]
                hit = np.zeros(len(q), bool)
                for d in docs:
                    hit |= reach_mask(vecs, scores, s.extra_vec(d.pid), k)
                same_article = (q["title"] == title_of[t.qid]).values
                counts.append(int(hit.sum()))
                per["target"].append(float(hit[s.row_of[t.qid]]))
                for g in ("nearby_passage", "nearby_article"):
                    m = ((q["target_qid"] == t.qid) & (q["set"] == g)).values
                    if m.any():
                        per[g].append(hit[m].mean())
                m = ((q["set"] == "broad").values & ~same_article)
                per["broad"].append(hit[m].mean())
                per["all_other"].append(hit[~same_article].mean())
            for g, vals in per.items():
                rows.append({"poison": kind, "k": k, "group": g, "unit": "poisons", "n": len(vals),
                             "ci": boot_mean(vals), "mean_questions_reached": np.mean(counts)})
    for d in s.broad:
        for k in C.KS:
            hit = reach_mask(vecs, scores, s.extra_vec(d.pid), k)
            for g, m in (("broad", (q["set"] == "broad").values), ("all_other", np.ones(len(q), bool))):
                rows.append({"poison": d.kind, "k": k, "group": g, "unit": "questions", "n": int(m.sum()),
                             "ci": boot_mean(hit[m].astype(float)), "mean_questions_reached": int(hit.sum())})
    return pd.DataFrame(rows)


def broad_margin(s: Study) -> pd.DataFrame:
    """How far the broad poisons sit from the top-k boundary: best rank they reach and similarity gap."""
    rows = []
    for d in s.broad:
        sim = s.all_qvecs @ s.extra_vec(d.pid)
        gap10 = sim - s.clean_scores[:, 9]
        rows.append({"poison": d.kind, "max_gap_to_rank10": float(gap10.max()),
                     "median_gap_to_rank10": float(np.median(gap10)),
                     "n_within_0.02_of_rank10": int((gap10 > -0.02).sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- model-based tables

def _per_target_ratio(d: pd.DataFrame, num: str, den: str, unit: str = "qid"):
    g = d.groupby(unit).agg(n=(num, "sum"), d=(den, "sum"))
    return boot_ratio(g["n"].values, g["d"].values)


def uptake_table(df: pd.DataFrame) -> pd.DataFrame:
    base = df[(df["phase"] == "main") & (df["defence"] == "none") & (df["rep"] == 0)]
    clean = base[base["cond"] == "clean"].set_index(["model", "qid", "k"])["correct"]
    rows = []
    for (model, cond, k), d in base[base["cond"].isin(["false_fact", "injection", "false_fact_x5"])
                                    & (base["qid"] == base["owner"])].groupby(["model", "cond", "k"]):
        d = d.copy()
        d["hit"] = d["in_topk"] & d["uptake"]
        d["chit"] = d["in_topk"] & d["canary"]
        cc = clean.reindex(pd.MultiIndex.from_arrays([d["model"], d["qid"], d["k"]])).values.astype(float)
        rows.append({"model": model, "poison": cond, "k": k, "n": len(d),
                     "retrieved": boot_mean(d["in_topk"].astype(float)),
                     "uptake_given_retrieved": _per_target_ratio(d, "hit", "in_topk"),
                     "canary_only": _per_target_ratio(d, "chit", "in_topk") if cond == "injection"
                     else (np.nan,) * 3,
                     "uptake_all": boot_mean(d["uptake"].astype(float)),
                     "acc_clean": boot_mean(cc), "acc_poisoned": boot_mean(d["correct"].astype(float)),
                     "acc_change": boot_diff(d["correct"].astype(float).values, cc)})
    return pd.DataFrame(rows)


def broad_uptake_table(df: pd.DataFrame) -> pd.DataFrame:
    base = df[(df["phase"] == "main") & (df["defence"] == "none") & (df["rep"] == 0)]
    rows = []
    for (model, cond), d in base[base["cond"].str.startswith("broad")].groupby(["model", "cond"]):
        hit = d[d["in_topk"]]
        rows.append({"model": model, "poison": cond, "n_questions": d["qid"].nunique(),
                     "n_retrieved": len(hit), "uptake_given_retrieved":
                     boot_mean(hit["uptake"].astype(float)) if len(hit) else (np.nan,) * 3})
    unrel = base[base["cond"].isin(["false_fact", "injection"]) & (base["qset"] == "broad")]
    for (model, cond), d in unrel.groupby(["model", "cond"]):
        rows.append({"model": model, "poison": cond + " on unrelated questions", "n_questions": len(d),
                     "n_retrieved": int(d["in_topk"].sum()),
                     "uptake_given_retrieved": boot_mean(d.loc[d["in_topk"], "uptake"].astype(float))})
    return pd.DataFrame(rows)


def spread_table(df: pd.DataFrame) -> pd.DataFrame:
    base = df[(df["defence"] == "none") & (df["rep"] == 0) & (df["k"] == C.MAIN_K)]
    clean = base[(base["phase"] == "main") & (base["cond"] == "clean")].set_index(["model", "qid"])["correct"]
    rows = []
    for phase, conds in (("main", ["false_fact", "injection", "false_fact_x5"]), ("hop2", ["note"])):
        sub = base[(base["phase"] == phase) & base["cond"].isin(conds)]
        sub = sub[(sub["qset"] != "broad")]
        for (model, cond, qset), d in sub.groupby(["model", "cond", "qset"]):
            d = d.copy()
            d["hit"] = d["in_topk"] & d["uptake"]
            cc = clean.reindex(pd.MultiIndex.from_arrays([d["model"], d["qid"]])).values.astype(float)
            rows.append({"model": model, "poison": cond, "set": qset, "n": len(d),
                         "retrieved": boot_mean(d["in_topk"].astype(float)),
                         "uptake_given_retrieved": _per_target_ratio(d, "hit", "in_topk", "owner"),
                         "uptake_all": boot_mean(d["uptake"].astype(float)),
                         "acc_change": boot_diff(d["correct"].astype(float).values, cc)})
    return pd.DataFrame(rows)


def defence_table(df: pd.DataFrame) -> pd.DataFrame:
    """Uptake (targets, k=5) and clean accuracy for each defence, paired with no defence on the same items."""
    k5 = df[(df["k"] == C.MAIN_K) & (df["rep"] == 0) & (df["phase"].isin(["main", "defence"]))]
    variants = {
        "none": k5[k5["defence"] == "none"],
        "instruction filter": k5[k5["defence"] == "filter"],
        "quote your source": k5[k5["defence"] == "quote"],
        "source agreement": with_agreement(k5[k5["defence"] == "none"]),
        "filter + agreement": with_agreement(k5[k5["defence"] == "filter"]),
    }
    rows = []
    for model in models_in(df):
        ref = variants["none"][variants["none"]["model"] == model]
        for name, v in variants.items():
            v = v[v["model"] == model]
            if v.empty:
                continue
            row = {"model": model, "defence": name}
            cl = v[v["cond"] == "clean"].set_index("qid")
            cl_ref = ref[ref["cond"] == "clean"].set_index("qid").loc[cl.index]
            row["clean_n"] = len(cl)
            row["clean_acc"] = boot_mean(cl["correct"].astype(float))
            row["clean_acc_ref"] = boot_mean(cl_ref["correct"].astype(float))
            row["clean_cost"] = boot_diff(cl_ref["correct"].astype(float).values, cl["correct"].astype(float).values)
            row["abstain"] = boot_mean(v.loc[v["cond"] == "clean", "abstain"].astype(float)) \
                if "abstain" in v else (0.0, 0.0, 0.0)
            for cond in ("false_fact", "injection", "false_fact_x5"):
                t = v[(v["cond"] == cond) & (v["qid"] == v["owner"])].copy()
                t["hit"] = t["in_topk"] & t["uptake"]
                row[cond] = _per_target_ratio(t, "hit", "in_topk")
                row[cond + "_n"] = int(t["in_topk"].sum())
            f = v[v["cond"].str.endswith("_forced")]
            row["broad_forced"] = boot_mean(f["uptake"].astype(float)) if len(f) else (np.nan,) * 3
            rows.append(row)
    return pd.DataFrame(rows)


def noise_table(df: pd.DataFrame) -> pd.DataFrame:
    rep = df[df["phase"] == "repeat"]
    key = ["model", "cond", "qid", "k", "owner", "defence"]
    base = df[(df["phase"] == "main") & (df["rep"] == 0) & (df["defence"] == "none")]
    m = rep.merge(base, on=key, suffixes=("_r1", "_r0"))
    rows = []
    for model, d in m.groupby("model"):
        clean = d[d["cond"] == "clean"]
        pois = d[d["cond"] != "clean"]
        rows.append({
            "model": model, "pairs": len(d),
            "output_differs": boot_mean((d["output_r1"] != d["output_r0"]).astype(float)),
            "answer_differs": boot_mean((d["answer_r1"] != d["answer_r0"]).astype(float)),
            "correct_flips": boot_mean((d["correct_r1"] != d["correct_r0"]).astype(float)),
            "uptake_flips": boot_mean((pois["uptake_r1"] != pois["uptake_r0"]).astype(float)),
            "clean_acc_r0": clean["correct_r0"].mean(), "clean_acc_r1": clean["correct_r1"].mean(),
            "clean_acc_gap": boot_diff(clean["correct_r1"].astype(float).values,
                                       clean["correct_r0"].astype(float).values),
            "uptake_r0": pois["uptake_r0"].mean(), "uptake_r1": pois["uptake_r1"].mean(),
        })
    return pd.DataFrame(rows)


def clean_accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    base = df[(df["phase"] == "main") & (df["cond"] == "clean") & (df["defence"] == "none") & (df["rep"] == 0)]
    rows = []
    for (model, k), d in base.groupby(["model", "k"]):
        for qset in ["all"] + sorted(d["qset"].unique()):
            dd = d if qset == "all" else d[d["qset"] == qset]
            if k != C.MAIN_K and qset == "all":
                continue
            rows.append({"model": model, "k": k, "set": qset, "n": len(dd),
                         "acc": boot_mean(dd["correct"].astype(float))})
    return pd.DataFrame(rows)


def false_positive_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Clean answers that already contain a target's made-up string: should be zero."""
    clean = df[(df["cond"] == "clean") & (df["defence"] == "none") & (df["phase"] == "main")]
    rows = []
    for model, d in clean.groupby("model"):
        own = d[d["owner"] != ""]
        rows.append({"model": model, "clean answers checked": len(own),
                     "contain a made-up string": int(own["adopted"].sum()), "clean answers": len(d),
                     "open with 'unavailable' or 'unknown'": int(sum(
                         gives_instructed_reply(a, "unavailable") or gives_instructed_reply(a, "unknown")
                         for a in d["answer"].fillna("")))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures

def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_reach(reach: pd.DataFrame, path):
    groups = ["target", "nearby_passage", "nearby_article", "all_other"]
    kinds = ["false_fact", "injection", "false_fact_x5", "broad_generic"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, g in zip(axes, groups):
        for color, kind in zip(PALETTE, kinds):
            d = reach[(reach["poison"] == kind) & (reach["group"] == g)].sort_values("k")
            if d.empty:
                continue
            m = np.array([c[0] for c in d["ci"]]) * 100
            lo = np.array([c[1] for c in d["ci"]]) * 100
            hi = np.array([c[2] for c in d["ci"]]) * 100
            ax.plot(d["k"], m, color=color, linewidth=2, marker="o", markersize=5, label=LABEL[kind])
            ax.fill_between(d["k"], lo, hi, color=color, alpha=0.12, linewidth=0)
        ax.set_title(SET_LABEL[g], fontsize=10, color=INK)
        ax.set_xticks(list(C.KS))
        ax.set_xlabel("k (passages retrieved)", fontsize=9, color=MUTED)
        _style(ax)
    axes[0].set_ylabel("questions reached (%)", fontsize=9, color=MUTED)
    axes[0].set_ylim(-3, 103)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Reach: share of questions whose top-k contains the planted passage", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _bars(ax, labels, series, models):
    x = np.arange(len(labels))
    w = 0.8 / max(1, len(models))
    for i, (model, color) in enumerate(zip(models, PALETTE)):
        vals = series[model]
        m = np.array([v[0] for v in vals]) * 100
        lo = m - np.array([v[1] for v in vals]) * 100
        hi = np.array([v[2] for v in vals]) * 100 - m
        ax.bar(x + (i - (len(models) - 1) / 2) * w, np.nan_to_num(m), w * 0.92, color=color, label=model,
               yerr=[np.nan_to_num(lo), np.nan_to_num(hi)], error_kw={"elinewidth": 1, "ecolor": MUTED,
                                                                         "capsize": 2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    _style(ax)


def fig_uptake(up: pd.DataFrame, dt: pd.DataFrame, models, path):
    kinds = ["false_fact", "false_fact_x5", "injection"]
    series = {}
    for m in models:
        vals = []
        for kind in kinds:
            d = up[(up["model"] == m) & (up["poison"] == kind) & (up["k"] == C.MAIN_K)]
            vals.append(d["uptake_given_retrieved"].iloc[0] if len(d) else (np.nan,) * 3)
        d = dt[(dt["model"] == m) & (dt["defence"] == "none")]
        vals.append(d["broad_forced"].iloc[0] if len(d) else (np.nan,) * 3)
        series[m] = vals
    fig, ax = plt.subplots(figsize=(8, 3.8))
    _bars(ax, ["false fact (1 copy)", "false fact (5 copies)", "injection", "broad, forced to rank 1"],
          series, models)
    ax.set_ylabel("uptake when retrieved (%)", fontsize=9, color=MUTED)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8, ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Uptake at k=5: answer repeats the planted claim or obeys the planted instruction",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_spread(sp: pd.DataFrame, up: pd.DataFrame, models, path):
    cats = [("main", "target"), ("false_fact", "nearby_passage"), ("false_fact", "nearby_article"),
            ("note", "target"), ("note", "nearby_passage"), ("note", "nearby_article")]
    labels = ["target\n(poison)", "same paragraph\n(poison)", "same article\n(poison)",
              "target\n(note, hop 2)*", "same paragraph\n(note, hop 2)*", "same article\n(note, hop 2)*"]
    series = {}
    for m in models:
        vals = []
        for cond, s in cats:
            if cond == "main":
                d = up[(up["model"] == m) & (up["poison"] == "false_fact") & (up["k"] == C.MAIN_K)]
                vals.append(d["uptake_all"].iloc[0] if len(d) else (np.nan,) * 3)
            else:
                d = sp[(sp["model"] == m) & (sp["poison"] == cond) & (sp["set"] == s)]
                vals.append(d["uptake_all"].iloc[0] if len(d) else (np.nan,) * 3)
        series[m] = vals
    fig, ax = plt.subplots(figsize=(10, 3.8))
    _bars(ax, labels, series, models)
    ax.set_ylabel("answers with the made-up string (%)", fontsize=9, color=MUTED)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8, ncol=len(models), loc="upper right")
    ax.set_title("Spread of one false-fact passage (k=5), and of the adopted answer written back as a note",
                 fontsize=10, color=INK)
    fig.text(0.01, 0.01, "* hop 2 covers only targets whose answer adopted the false fact; the poison is removed "
             "and the note is the only planted text.", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_defences(dt: pd.DataFrame, models, path):
    names = ["none", "instruction filter", "quote your source", "source agreement", "filter + agreement"]
    labels = ["none", "instruction\nfilter", "quote your\nsource", "source\nagreement", "filter +\nagreement"]
    panels = [("clean_cost", "clean accuracy lost (pp)"), ("false_fact", "false-fact uptake when retrieved (%)"),
              ("injection", "injection uptake when retrieved (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    for ax, (col, title) in zip(axes, panels):
        series = {}
        for m in models:
            vals = []
            for n in names:
                d = dt[(dt["model"] == m) & (dt["defence"] == n)]
                vals.append(d[col].iloc[0] if len(d) else (np.nan,) * 3)
            series[m] = vals
        _bars(ax, labels, series, models)
        ax.set_title(title, fontsize=10, color=INK)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        if col != "clean_cost":
            ax.set_ylim(0, 105)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Defences at k=5: what each one stops and what it costs on clean questions", fontsize=11,
                 color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- report

def _md(df: pd.DataFrame) -> str:
    if "model" in df.columns:
        rank = {m: i for i, m in enumerate(MODEL_ORDER)}
        df = df.assign(_r=df["model"].map(lambda m: rank.get(m, 99))).sort_values("_r", kind="stable")
        df = df.drop(columns="_r")
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def _pp(ci) -> str:
    m, lo, hi = ci
    return "n/a" if np.isnan(m) else f"{100 * m:+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}]"


def _pct(ci, digits=0) -> str:
    m, lo, hi = ci
    if np.isnan(m):
        return "n/a"
    return f"{100 * m:.{digits}f}% [{100 * lo:.{digits}f}, {100 * hi:.{digits}f}]"


def headline(reach, up, sp, dt, nz, models) -> str:
    def r(kind, group, k=C.MAIN_K):
        d = reach[(reach["poison"] == kind) & (reach["group"] == group) & (reach["k"] == k)]
        return d.iloc[0]

    ff_t = r("false_fact", "target")
    ff_np, ff_na, ff_u = r("false_fact", "nearby_passage"), r("false_fact", "nearby_article"), r("false_fact", "all_other")
    bg, bt = r("broad_generic", "all_other"), r("broad_terms", "all_other")
    bg10 = r("broad_generic", "all_other", 10)
    out = [f"- **Reach (k=5).** A false-fact passage that restates its target question is in the top 5 for "
           f"{_pct(ff_t['ci'])} of the 48 targets, for {_pct(ff_np['ci'])} of nearby questions answered by the same "
           f"paragraph, {_pct(ff_na['ci'])} of other questions on the same article, and {_pct(ff_u['ci'], 2)} of "
           f"questions on other articles: on average it reaches {ff_t['mean_questions_reached']:.1f} of the 10,570 "
           f"validation questions, its target included. The keyword-stuffed passages reach {int(bg['mean_questions_reached'])} and "
           f"{int(bt['mean_questions_reached'])} of 10,570 at k=5 ({int(bg10['mean_questions_reached'])} for the "
           "generic one at k=10)."]
    for m in models:
        u = up[(up["model"] == m) & (up["k"] == C.MAIN_K)].set_index("poison")
        d = dt[dt["model"] == m].set_index("defence")
        parts = []
        if "false_fact" in u.index:
            parts.append(f"false fact {_pct(u.loc['false_fact', 'uptake_given_retrieved'])}")
        if "false_fact_x5" in u.index:
            parts.append(f"five copies {_pct(u.loc['false_fact_x5', 'uptake_given_retrieved'])}")
        if "injection" in u.index:
            parts.append(f"injection {_pct(u.loc['injection', 'uptake_given_retrieved'])}")
        if "none" in d.index:
            parts.append(f"broad passage forced to rank 1 {_pct(d.loc['none', 'broad_forced'])}")
        hop = sp[(sp["model"] == m) & (sp["poison"] == "note") & (sp["set"] == "target")]
        line = f"- **{m}** uptake when retrieved (k=5): " + "; ".join(parts) + "."
        if len(hop):
            line += (f" Second hop: the adopted answer, written back as a note with the poison removed, "
                     f"is repeated on {_pct(hop.iloc[0]['uptake_all'])} of the {hop.iloc[0]['n']} targets that "
                     "adopted it.")
        out.append(line)
        if "none" in d.index:
            def cost(n):
                return _pp(d.loc[n, "clean_cost"]) if n in d.index else "n/a"
            seg = []
            if "instruction filter" in d.index:
                seg.append(f"instruction filter: injection {_pct(d.loc['none', 'injection'])} -> "
                           f"{_pct(d.loc['instruction filter', 'injection'])}, false fact "
                           f"{_pct(d.loc['instruction filter', 'false_fact'])}, clean cost {cost('instruction filter')}")
            if "quote your source" in d.index:
                seg.append(f"quote prompt: false fact -> {_pct(d.loc['quote your source', 'false_fact'])}, injection -> "
                           f"{_pct(d.loc['quote your source', 'injection'])}, clean cost {cost('quote your source')}")
            if "source agreement" in d.index:
                seg.append(f"source agreement: false fact -> {_pct(d.loc['source agreement', 'false_fact'])}, five "
                           f"copies -> {_pct(d.loc['source agreement', 'false_fact_x5'])}, clean cost "
                           f"{cost('source agreement')} (abstains on {_pct(d.loc['source agreement', 'abstain'])} of clean "
                           "questions)")
            out.append("  - defences: " + "; ".join(seg) + ".")
    if len(nz):
        out.append("- **Noise.** Repeating identical calls changed " + "; ".join(
            f"{r_['model']} {_pct(r_['output_differs'], 1)} of {r_['pairs']} outputs"
            for _, r_ in nz.set_index("model").loc[[m for m in models if m in set(nz["model"])]].reset_index().iterrows())
            + ".")
    return "\n".join(out)


def analyze() -> None:
    from .handcheck import agreement_report
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    s = Study()
    df = load_rows()
    models = models_in(df)
    reach = reach_table(s)
    margin = broad_margin(s)
    up = uptake_table(df)
    bu = broad_uptake_table(df)
    sp = spread_table(df)
    dt = defence_table(df)
    nz = noise_table(df)
    ca = clean_accuracy_table(df)
    fp = false_positive_rate(df)
    hc = agreement_report()
    metas = {}
    for p in sorted(C.RESULTS.glob("*/run_meta.json")):
        meta = json.loads(p.read_text())
        metas[meta["model"]] = meta

    fig_reach(reach, C.FIGURES / "reach_by_k.png")
    fig_uptake(up, dt, models, C.FIGURES / "uptake.png")
    fig_spread(sp, up, models, C.FIGURES / "spread.png")
    fig_defences(dt, models, C.FIGURES / "defence_tradeoff.png")

    head = headline(reach, up, sp, dt, nz, models)
    (C.REPORTS / "headline.md").write_text(head + "\n")
    L = ["# Results", "",
         "Generated by `uv run poison analyze` from `results/*/rows.jsonl` and the cached embeddings. "
         "Every number below is computed by code; 95% intervals are percentile bootstraps "
         "(2,000 resamples) over the unit named in each table.", "",
         "> **Limits.** Small local models (8B and 12B, 4-bit quantised by Ollama; the planned 30B model was not run), one corpus (10,000 SQuAD "
         "paragraphs, Wikipedia 2016), one retriever (bge-small), templated poisons, 48 targets. SQuAD questions "
         "were written while reading the paragraph, so they share its wording, which makes retrieval easier "
         "than for real user questions. Uptake is judged by string match on made-up answers and canary codes; "
         "a labelled sample of 90 outputs checks this (section 7).", "", "## Headline", "",
         "Intervals are 95% bootstrap intervals; `pp` is percentage points.", "", head, ""]

    # 1. reach
    L += ["## 1. Reach (retrieval only, no model)", "",
          "Share of questions whose top-k contains the planted passage. Targeted poisons: mean over the 48 "
          "poisons (one planted at a time), interval over poisons. `mean questions reached` counts, out of all "
          "10,570 validation questions, how many have the passage in their top-k. Broad poisons: one passage, "
          "interval over questions.", ""]
    t = reach.copy()
    t["poison"] = t["poison"].map(LABEL)
    t["group"] = t["group"].map(SET_LABEL)
    t["reach"] = t["ci"].map(lambda c: fmt(c, digits=2))
    t["mean questions reached"] = t["mean_questions_reached"].map(lambda v: f"{v:.1f}")
    L += [_md(t[["poison", "k", "group", "n", "reach", "mean questions reached"]]), ""]
    mg = margin.copy()
    mg["poison"] = mg["poison"].map(LABEL)
    L += ["Distance of the broad poisons from the top-10 boundary (cosine similarity minus the 10th clean "
          "score; positive means retrieved):", "", _md(mg.round(4)), "",
          "![reach](figures/reach_by_k.png)", ""]

    # 2. clean accuracy
    t = ca.copy()
    t["acc"] = t["acc"].map(fmt)
    L += ["## 2. Clean accuracy (no poison)", "",
          "Correct = a gold SQuAD answer appears in the model's short answer after normalisation.", "",
          _md(t), ""]

    # 3. uptake
    t = up.copy()
    t["poison"] = t["poison"].map(LABEL)
    for c in ("retrieved", "uptake_given_retrieved", "canary_only", "uptake_all", "acc_clean", "acc_poisoned"):
        t[c] = t[c].map(fmt)
    t["acc_change"] = t["acc_change"].map(_pp)
    L += ["## 3. Uptake on the target question", "",
          "One poison per target, planted alone. `uptake when retrieved` = answers that contain the made-up "
          "string (false fact), or outputs that follow the injection (the canary code appears, or the answer "
          "opens with the instructed reply 'unavailable'), among targets whose top-k holds the poison; "
          "interval over targets. `canary only` counts just the outputs that contain the code. `acc change` is poisoned minus clean accuracy on the "
          "same targets.", "",
          _md(t.rename(columns={"uptake_given_retrieved": "uptake when retrieved", "uptake_all": "uptake (all)",
                                "canary_only": "canary only", "acc_clean": "acc clean", "acc_poisoned": "acc poisoned",
                                "acc_change": "acc change"})), "",
          "![uptake](figures/uptake.png)", ""]
    t = bu.copy()
    t["poison"] = t["poison"].map(lambda p: LABEL.get(p, p))
    t["uptake_given_retrieved"] = t["uptake_given_retrieved"].map(fmt)
    L += ["Broad poisons (all 783 questions at k=5; the `forced rank 1` rows put the passage first in the "
          "top 5 for 100 broad questions regardless of retrieval) and targeted poisons reaching unrelated "
          "questions:", "", _md(t.rename(columns={"uptake_given_retrieved": "uptake when retrieved"})), ""]
    fpt = fp.copy()
    L += ["Checker false positives on clean answers: answers that already contain their target's made-up "
          "string, and answers that already open with the reply word an instruction asks for:", "",
          _md(fpt), ""]

    # 4. spread
    t = sp.copy()
    t["poison"] = t["poison"].map(LABEL)
    t["set"] = t["set"].map(SET_LABEL)
    for c in ("retrieved", "uptake_given_retrieved", "uptake_all"):
        t[c] = t[c].map(fmt)
    t["acc_change"] = t["acc_change"].map(_pp)
    n_notes = df[(df["phase"] == "hop2") & (df["qset"] == "target")].groupby("model").size()
    L += ["## 4. Spread", "",
          "Nearby questions are other real SQuAD questions on the target's article: `same paragraph` are "
          "answered by the target's paragraph, `same article` by its other paragraphs. The second hop takes "
          "each target whose answer adopted the false fact, writes that answer back into the corpus as a "
          "short note (the poison itself removed), and asks the target and its nearby questions again. "
          "`uptake when retrieved` interval over targets; `acc change` is against the clean run.", "",
          "Notes written back (targets that adopted the false fact at k=5): "
          + ", ".join(f"{m}: {int(n_notes.get(m, 0))}" for m in models), "",
          _md(t.rename(columns={"uptake_given_retrieved": "uptake when retrieved", "uptake_all": "uptake (all)",
                                "acc_change": "acc change"})), "",
          "![spread](figures/spread.png)", ""]

    # 5. defences
    t = dt.copy()
    out = pd.DataFrame({
        "model": t["model"], "defence": t["defence"], "clean n": t["clean_n"],
        "clean acc": t["clean_acc"].map(fmt), "clean cost (pp)": t["clean_cost"].map(_pp),
        "abstain (clean)": t["abstain"].map(fmt),
        "false fact uptake": t["false_fact"].map(fmt), "n": t["false_fact_n"],
        "false fact x5 uptake": t["false_fact_x5"].map(fmt),
        "injection uptake": t["injection"].map(fmt), "n ": t["injection_n"],
        "broad forced uptake": t["broad_forced"].map(fmt)})
    L += ["## 5. Defences", "",
          "All at k=5. Uptake is among targets whose top-5 holds the poison (interval over targets). "
          "Clean cost = accuracy without the defence minus with it, on the same clean questions (paired "
          "bootstrap); positive means the defence loses accuracy. The quote prompt runs on targets, nearby "
          f"and {C.QUOTE_BROAD} of the 500 broad questions (reduced profile: targets and nearby only), so its "
          "clean n is smaller. Source agreement is a "
          "post-hoc rule on the undefended answers: an answer survives only if at least two of the shown "
          "passages contain all its content words, otherwise the system abstains (abstentions count as "
          "wrong). The cross-encoder reranker was not run: no reranker model was available locally, and the "
          "study's download budget was one embedding model.", "", _md(out), "",
          "![defences](figures/defence_tradeoff.png)", ""]

    # 6. noise
    t = nz.copy()
    if t.empty:
        t = pd.DataFrame({"status": ["repeat phase not run yet"]})
    else:
        for c in ("output_differs", "answer_differs", "correct_flips", "uptake_flips"):
            t[c] = t[c].map(fmt)
        t["clean_acc_gap"] = t["clean_acc_gap"].map(_pp)
        for c in ("clean_acc_r0", "clean_acc_r1", "uptake_r0", "uptake_r1"):
            t[c] = t[c].map(lambda v: f"{100 * v:.1f}%")
    L += ["## 6. Run-to-run noise", "",
          "Identical requests (same model, prompt, temperature 0, seed) sent a second time with the cache "
          f"bypassed: the clean condition on targets, nearby and {C.REPEAT_BROAD} broad questions (reduced-profile "
          "models: targets only), plus the single-copy false fact and injection on targets.", "", _md(t), ""]

    # 7. hand check
    L += ["## 7. Checker against labels", "", hc, ""]

    # 8. runs
    rows = []
    for m in models:
        meta = metas.get(m, {})
        calls = df[df["model"] == m].drop_duplicates("call")["wall"].astype(float)
        ok = calls[calls < 60]
        rows.append({"model": m, "profile": meta.get("profile"), "phases": ", ".join(meta.get("phases_done", [])),
                     "digest": meta.get("digest", "")[:19], "think": meta.get("think"),
                     "rows": len(df[df["model"] == m]), "distinct model calls": len(calls),
                     "median s per call": round(float(ok.median()), 2),
                     "model time (h)": round(float(ok.sum()) / 3600, 2),
                     "stalled calls (>60 s)": int((calls >= 60).sum())})
    L += ["## 8. Runs", "", f"Options for every call: `{json.dumps(C.LLM_OPTIONS)}`. `think: false` is sent to "
          "models that report the thinking capability (the qwen3 models); others get no think field. "
          "Model time sums the wall-clock time of each distinct call when it was first made, leaving out calls "
          "that took over 60 s (the laptop slept during part of the run, which stretched a few calls to "
          "15 minutes). For part of the run two models shared the GPU, which slows each call.", "",
          _md(pd.DataFrame(rows)), ""]
    (C.REPORTS / "results.md").write_text("\n".join(L))
    print(f"wrote {C.REPORTS / 'results.md'} and {len(list(C.FIGURES.glob('*.png')))} figures")
