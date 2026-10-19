"""uv run poison <command>"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def cmd_fetch(args):
    from . import config as C
    from . import data
    for split, info in data.fetch(force=args.force).items():
        print(split, info)
    from .embed import Embedder
    Embedder().model  # downloads the pinned embedding model into the Hugging Face cache
    print("embedding model", C.EMBED_MODEL, "@", C.EMBED_REVISION)


def cmd_prepare(args):
    from . import config as C
    from .study import Study
    s = Study()
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    cols = ["qid", "set", "target_qid", "title", "question", "answers", "gold_pid", "gold_rank"]
    s.selected[cols].assign(answers=lambda d: d["answers"].map(json.dumps)).to_csv(
        C.RESULTS / "question_sets.csv", index=False)
    derived = C.DATA / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    s.corpus.to_parquet(derived / "corpus.parquet", index=False)
    with open(derived / "planted.jsonl", "w") as f:
        for d in s.all_planted():
            f.write(json.dumps(d.__dict__) + "\n")
    print(f"corpus {len(s.corpus)} passages; question sets {s.selected['set'].value_counts().to_dict()}")
    print(f"planted documents built: {len(s.all_planted())}")


def cmd_run(args):
    from .experiment import run
    from .study import Study
    s = Study()
    for model in args.model:
        run(s, model, phases=tuple(args.phases.split(",")), reduced=args.reduced, cache_only=args.cache_only)


def cmd_analyze(args):
    from .analysis import analyze
    analyze()


def cmd_handcheck(args):
    from .handcheck import write_sample
    write_sample(args.model or ["qwen3:8b", "llama3.1:8b"], args.n)


def main(argv=None):
    p = argparse.ArgumentParser(prog="poison", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download SQuAD v1.1 and the embedding model")
    f.add_argument("--force", action="store_true")
    f.set_defaults(func=cmd_fetch)
    sub.add_parser("prepare", help="sample the corpus, choose questions, build planted documents"
                   ).set_defaults(func=cmd_prepare)
    r = sub.add_parser("run", help="run every condition for one or more models (cached)")
    r.add_argument("--model", action="append", required=True)
    r.add_argument("--phases", default="main,hop2,defence,repeat")
    r.add_argument("--reduced", action="store_true",
                   help="smaller broad set and k=5 only, for the slower models")
    r.add_argument("--cache-only", action="store_true",
                   help="never call a model; phases with uncached calls are left out")
    r.set_defaults(func=cmd_run)
    sub.add_parser("analyze", help="write reports/results.md and figures").set_defaults(func=cmd_analyze)
    h = sub.add_parser("handcheck", help="draw the sample of outputs for hand labelling")
    h.add_argument("--model", action="append")
    h.add_argument("--n", type=int, default=60)
    h.set_defaults(func=cmd_handcheck)
    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
