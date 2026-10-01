"""Runnable interface (Section 10) - CLI.

Istemal:
  python cli.py --image samples/doliprane.jpg --json product.json
  python cli.py --url https://.../img.jpg --json product.json --out result.json
  python cli.py --json '{"id":"1","name":"Doliprane 1000 mg"}' --base64 <b64>
  python cli.py serve --port 8000        # FastAPI server start
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def cmd_analyze(args: argparse.Namespace) -> int:
    from app.agent import ProductQualityAgent
    from app.input_loader import InputError

    agent = ProductQualityAgent(config_path=args.config)
    try:
        report = agent.analyze(
            args.image,
            args.json,
            base64_str=args.base64,
            url=args.url,
        )
    except InputError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Saved -> {args.out}")
    else:
        print(text)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Product Image Anomaly Detection & Quality Agent")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="ek image analyze karo (default)")
    run.add_argument("--image", help="image file path")
    run.add_argument("--url", help="image URL")
    run.add_argument("--base64", help="base64 encoded image")
    run.add_argument("--json", required=True, help="product JSON: file path ya inline JSON string")
    run.add_argument("--out", help="result JSON yahan save karo")
    run.add_argument("--config", default=None, help="config.yaml ka path")
    run.set_defaults(func=cmd_analyze)

    serve = sub.add_parser("serve", help="FastAPI server chalao")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--config", default=None, help="config.yaml ka path")
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    known = {"run", "serve", "-h", "--help"}
    if argv and argv[0] not in known:
        argv.insert(0, "run")  # default subcommand
    args = build_parser().parse_args(argv)
    if args.command is None:
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
