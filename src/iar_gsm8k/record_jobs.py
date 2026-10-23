from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .data import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--jobs-json", required=True)
    parser.add_argument("--filename", default="slurm_jobs.json")
    args = parser.parse_args()
    jobs = json.loads(args.jobs_json)
    write_json(
        args.run_dir.resolve() / args.filename,
        {
            "run_id": args.run_id,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "jobs": jobs,
        },
    )


if __name__ == "__main__":
    main()
