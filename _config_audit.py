"""Config-consistency audit for the LD3 repo (read-only).

Three checks:
  1. Every config/*.yaml|yml parses and loads through config.params.load_config
     (extends chains resolved; unknown keys rejected by the loader itself).
  2. Configs never referenced by name anywhere in tools/ scripts/ tests/ docs/
     paper/ uav_isac/  -> dead configs (possible cruft).
  3. References to config/<name>.yaml that do not exist in the tree -> dead
     references (possible broken experiment pointers).

Usage:  pytrch_ven\\Scripts\\python.exe _config_audit.py
"""

import os
import re
import sys

ROOT = "D:/BYLW/LD3"
CONFIG_DIR = os.path.join(ROOT, "config")
SEARCH_DIRS = ["tools", "scripts", "tests", "docs", "paper", "uav_isac", "frozen", "advice"]

sys.path.insert(0, ROOT)


def main():
    configs = sorted(
        f
        for f in os.listdir(CONFIG_DIR)
        if f.endswith((".yaml", ".yml"))
    )

    # --- 1. loadability --------------------------------------------------
    from config.params import load_config

    load_fail = []
    for name in configs:
        path = os.path.join(CONFIG_DIR, name)
        try:
            load_config(path)
        except Exception as e:
            load_fail.append((name, type(e).__name__, str(e)[:200]))

    print(f"== configs total: {len(configs)}")
    print(f"== load failures: {len(load_fail)}")
    for name, exc, msg in load_fail:
        print(f"  FAIL {name}: {exc}: {msg}")

    # --- 2. dead configs (no textual reference anywhere) ----------------
    refs = {name: 0 for name in configs}
    for sd in SEARCH_DIRS:
        base = os.path.join(ROOT, sd)
        if not os.path.isdir(base):
            continue
        for dirpath, _dirs, files in os.walk(base):
            for fn in files:
                if not fn.endswith((".md", ".py", ".yaml", ".yml", ".json", ".txt", ".ps1")):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    txt = open(p, encoding="utf-8", errors="ignore").read()
                except Exception:
                    continue
                for name in configs:
                    stem = name[:-5]  # without .yaml/.yml
                    if stem in txt or ("config/" + name) in txt or ("config\\" + name) in txt:
                        refs[name] += 1

    dead = sorted(name for name, c in refs.items() if c == 0)
    print(f"\n== dead configs (never referenced by name in tree): {len(dead)}")
    for name in dead:
        print(f"  {name}")

    # --- 3. dead references ---------------------------------------------
    pat = re.compile(r"config[/\\]([A-Za-z0-9_.\-]+\.ya?ml)")
    missing = set()
    hits = {}
    for sd in SEARCH_DIRS:
        base = os.path.join(ROOT, sd)
        if not os.path.isdir(base):
            continue
        for dirpath, _dirs, files in os.walk(base):
            for fn in files:
                if not fn.endswith((".md", ".py", ".yaml", ".yml", ".json", ".txt", ".ps1")):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    txt = open(p, encoding="utf-8", errors="ignore").read()
                except Exception:
                    continue
                for m in pat.finditer(txt):
                    name = m.group(1)
                    if not os.path.isfile(os.path.join(CONFIG_DIR, name)):
                        missing.add(name)
                        hits.setdefault(name, []).append(
                            os.path.relpath(p, ROOT).replace("\\", "/")
                        )

    print(f"\n== references to config files that do not exist: {len(missing)}")
    for name in sorted(missing):
        print(f"  {name}  <- {hits[name][:4]}")


if __name__ == "__main__":
    main()