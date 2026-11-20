"""Preview or explicitly delete old prediction telemetry."""
from __future__ import annotations

import argparse
import json

try:
    from .prediction_store import cleanup_telemetry
except ImportError:  # python mlops/cleanup_telemetry.py
    from prediction_store import cleanup_telemetry


def main(argv=None):
    parser = argparse.ArgumentParser(description="Preview or delete old MLOps prediction telemetry")
    parser.add_argument("--older-than-days", type=int, required=True,
                        help="Only match telemetry older than this many days")
    parser.add_argument("--apply", action="store_true",
                        help="Delete matching rows; without this flag the command is a dry run")
    args = parser.parse_args(argv)
    result = cleanup_telemetry(args.older_than_days, apply=args.apply)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
