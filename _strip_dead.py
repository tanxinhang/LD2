"""P1b-1 mechanical dead-symbol removal (read-only audit, then rewrite).

Removes whole functions/classes that have ZERO callers anywhere in the tree
(verified by repo-wide grep on 2026-08-25 audit).  No behavior change: each
removed symbol is either an entry-only leaf or an explicitly-dead branch.

Deletion list (file, start_line, end_line inclusive, lines are 1-based as
shown by the read tool):
  uav_isac/agents/networks.py               2779-2834   GATEncoder (class, end of file)
  uav_isac/agents/networks.py               1030-1136   _parse_one (dead branch; caller
                                                        rewritten separately)
  uav_isac/agents/mappo_agent.py             350-386    act_batch (incl. trailing blank)
  uav_isac/coordination/hyperedge.py         353-432    nearfield_focus_targets (incl. blank)
  uav_isac/evaluation/shadow_horizon_router.py 277-283  shadow_reason_counts (end of file)
  uav_isac/evaluation/quantized_evidence_audit.py 393-458 summarize_quantized_method (incl. blanks)

Usage:  pytrch_ven\\Scripts\\python.exe _strip_dead.py
"""

import os

ROOT = "D:/BYLW/LD3"

DELETIONS = [
    ("uav_isac/agents/networks.py", 2779, 2834),
    ("uav_isac/agents/networks.py", 1030, 1136),
    ("uav_isac/agents/mappo_agent.py", 350, 386),
    ("uav_isac/coordination/hyperedge.py", 353, 432),
    ("uav_isac/evaluation/shadow_horizon_router.py", 277, 283),
    ("uav_isac/evaluation/quantized_evidence_audit.py", 393, 458),
]


def main() -> None:
    # group by file, delete higher lines first inside each file
    by_file = {}
    for rel, start, end in DELETIONS:
        by_file.setdefault(rel, []).append((start, end))
    for rel, ranges in sorted(by_file.items()):
        path = os.path.join(ROOT, rel)
        with open(path, "r", encoding="utf-8-sig") as fh:
            lines = fh.readlines()
        removed = 0
        for start, end in sorted(ranges, reverse=True):
            end = min(end, len(lines))
            if start < 1 or start > end:
                raise ValueError(f"bad range {rel}:{start}-{end}")
            del lines[start - 1:end]
            removed += end - start + 1
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.writelines(lines)
        print(f"stripped {removed} lines from {rel}")


if __name__ == "__main__":
    main()