"""Command line interface for TwoOfUsTunes."""

from __future__ import annotations

import argparse
from pathlib import Path

from engine.recommender import recommend, rerank_two_users
from engine.state_estimator import estimate


RESULTS_DIR = Path("results")


def main() -> None:
    parser = argparse.ArgumentParser(description="TwoOfUsTunes CLI")
    sub = parser.add_subparsers(dest="cmd")

    r1 = sub.add_parser("recommend", help="Recommend tracks for one user")
    r1.add_argument("--v", type=float, required=True, help="Valence value (-1..1)")
    r1.add_argument("--a", type=float, required=True, help="Arousal value (-1..1)")
    r1.add_argument("--topk", type=int, default=20, help="Number of tracks")
    r1.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "v1_rec.csv",
        help="Output CSV path",
    )

    r2 = sub.add_parser("recommend-two", help="Recommend tracks for two users")
    r2.add_argument("--v1", type=float, required=True, help="User A valence")
    r2.add_argument("--a1", type=float, required=True, help="User A arousal")
    r2.add_argument("--v2", type=float, required=True, help="User B valence")
    r2.add_argument("--a2", type=float, required=True, help="User B arousal")
    r2.add_argument(
        "--strategy",
        choices=["avg", "intersect", "union"],
        default="avg",
        help="Blend strategy",
    )
    r2.add_argument("--w", type=float, default=0.5, help="Weight for avg strategy")
    r2.add_argument("--topk", type=int, default=20, help="Number of tracks")
    r2.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "v2_rec.csv",
        help="Output CSV path",
    )

    sub.add_parser("estimate", help="Estimate emotion from context")

    args = parser.parse_args()

    if args.cmd == "recommend":
        items = recommend(state={"v": args.v, "a": args.a}, topk=args.topk)
        _persist_results(items, args.out)
    elif args.cmd == "recommend-two":
        items = rerank_two_users(
            e_a=(args.v1, args.a1),
            e_b=(args.v2, args.a2),
            strategy=args.strategy,
            w=args.w,
            topk=args.topk,
        )
        _persist_results(items, args.out)
    elif args.cmd == "estimate":
        # Placeholder example inputs until sensing modules are wired in
        result = estimate({"slider": {"v": 0.0, "a": 0.0}})
        print(result)  # noqa: T201 - simple CLI output
    else:
        parser.print_help()


def _persist_results(items, destination: Path) -> None:
    """Persist recommendation result to CSV on disk."""
    raise NotImplementedError("Serialize 'items' to CSV at 'destination'")


if __name__ == "__main__":  # pragma: no cover
    main()
