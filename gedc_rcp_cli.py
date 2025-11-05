#!/usr/bin/env python3
"""
Command-line wrapper for the GEDC reasoning co‑processor.

Usage examples:
  python gedc_rcp_cli.py --task '{"type": "bound", "which": "tr", "x": 1}' --pretty
  python gedc_rcp_cli.py --file task.json
  echo '{"type": "verify", ...}' | python gedc_rcp_cli.py --pretty
"""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

from gedc_rcp import solve  # expects gedc_rcp.py and gedc_core.py in the same directory


def main() -> None:
    parser = ArgumentParser(description="GEDC reasoning co‑processor CLI")
    parser.add_argument("--task", help="Task JSON string")
    parser.add_argument("--file", help="Path to a JSON file containing the task")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the output")
    args = parser.parse_args()

    if args.task:
        try:
            task = json.loads(args.task)
        except Exception as e:
            print(f"Could not parse JSON from --task: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.file:
        task_path = Path(args.file)
        try:
            task = json.loads(task_path.read_text())
        except Exception as e:
            print(f"Could not read task file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # read from stdin
        try:
            task = json.loads(sys.stdin.read())
        except Exception as e:
            print(f"Could not parse JSON from stdin: {e}", file=sys.stderr)
            sys.exit(1)

    # Run the task through the solver. Wrap in try/except so CLI never crashes.
    try:
        result = solve(task)
    except Exception as e:
        result = {"status": "error", "error": str(e)}

    if args.pretty:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
