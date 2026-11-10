"""dartbot CLI.

  python -m dartbot analyze gameplay.mov          # video -> data/throws.jsonl
  python -m dartbot fit                           # data/*.jsonl -> model.json
  python -m dartbot check                         # one live frame: canvas, player, wind, score
  python -m dartbot play [--dry-run]              # play live
"""
from __future__ import annotations

import argparse
from pathlib import Path

DATA = Path("data")
VIDEO_LOG = DATA / "video_throws.jsonl"
LIVE_LOG = DATA / "live_throws.jsonl"
MODEL = Path("model.json")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dartbot")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="extract throws from a gameplay recording")
    a.add_argument("video", type=Path)
    a.add_argument("--out", type=Path, default=VIDEO_LOG)

    f = sub.add_parser("fit", help="fit model.json from recorded throws")
    f.add_argument("logs", type=Path, nargs="*", default=[VIDEO_LOG, LIVE_LOG])
    f.add_argument("--model", type=Path, default=MODEL)

    c = sub.add_parser("check", help="detect everything on the current screen once")
    c.add_argument("--model", type=Path, default=MODEL)
    c.add_argument("--monitor", type=int)

    p = sub.add_parser("play", help="play the minigame")
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--log", type=Path, default=LIVE_LOG)
    p.add_argument("--monitor", type=int)
    p.add_argument("--dry-run", action="store_true", help="track and plan, never click")
    p.add_argument("--replay", type=Path, help="dry-run against a recording played in real time")
    p.add_argument("--throws", type=int, default=0, help="stop after N throws (0 = until game over)")
    p.add_argument("--streak", type=int, default=0,
                   help="stop after this many bullseyes in a row (0 = keep playing)")
    p.add_argument("--direction", choices=["up", "down", "any"], default="up",
                   help="click only while the dart swings this way (live data: up hits far more often)")
    p.add_argument("--debug", action="store_true",
                   help="verbose console + logs/<session>/dartbot.log and snapshots of odd moments")

    args = ap.parse_args(argv)

    if args.cmd == "analyze":
        from . import records
        from .analyze import analyze_video
        throws = analyze_video(args.video)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("")
        for t in throws:
            records.append(args.out, t)
        print(f"{len(throws)} throws -> {args.out}")

    elif args.cmd == "fit":
        from . import records
        from .physics import Model, fit_model
        throws = records.load(*args.logs)
        base = Model.load(args.model) if args.model.exists() else Model()
        model = fit_model(throws, base)
        model.save(args.model)
        print(f"model -> {args.model}")

    elif args.cmd == "check":
        from .bot import check
        check(args)

    elif args.cmd == "play":
        from .bot import play
        play(args)


if __name__ == "__main__":
    main()
