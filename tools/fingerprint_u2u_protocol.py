"""Write the deterministic U2U protocol/config implementation fingerprint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_isac.coordination.protocol_fingerprint import (
    fingerprint_protocol_implementation,
)


def fingerprint_document(config_path: Path) -> dict[str, object]:
    result = fingerprint_protocol_implementation(
        config_path, workspace_root=ROOT)
    return {
        "schema_version": result.schema_version,
        "sha256": result.sha256,
        "config_chain": list(result.config_chain),
        "files": [
            {
                "path": item.path,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
            for item in result.files
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = fingerprint_document(args.config)
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
