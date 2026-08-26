"""AST-based import-dependency auditor for uav_isac (read-only).

Outputs:
  1. The full import graph (module -> direct external deps) restricted to
     uav_isac internal edges plus first-party config/package edges.
  2. Modules imported by NO OTHER uav_isac/tools/scripts/test module that
     still exist, i.e. entry/dead leaves.
  3. The transitive closure of env_core.py and trainer.py (the deployment
     spine) vs modules NOT in that closure but present (research/tool-only
     candidates).
  4. Strongly connected components (import cycles) among uav_isac modules.

Usage:  pytrch_ven\\Scripts\\python.exe _dep_audit.py
"""

import ast
import os

ROOT = "D:/BYLW/LD3"
PKG = os.path.join(ROOT, "uav_isac")

# Audit 2026-08-25: scan for UTF-8 BOM python files (ast.parse chokes on
# them; some toolchains silently mis-read them).
import glob as _glob
bom_files = []
for path in _glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True):
    try:
        with open(path, "rb") as fh:
            if fh.read(3) == b"\xef\xbb\xbf":
                bom_files.append(os.path.relpath(path, ROOT))
    except OSError:
        pass
if bom_files:
    print("== UTF-8 BOM python files (hygiene finding):", len(bom_files))
    for p in sorted(bom_files):
        print("   ", p)
else:
    print("== UTF-8 BOM python files: none")

# Map every uav_isac module path to its python package name.
module_names = {}
for dirpath, _dirs, files in os.walk(PKG):
    for fn in files:
        if fn.endswith(".py") and fn != "__init__.py":
            rel = os.path.relpath(os.path.join(dirpath, fn), ROOT)
            mod = rel[:-3].replace(os.sep, ".")
            module_names[mod] = os.path.join(dirpath, fn)

# The configuration root is first-party even though it lives outside uav_isac.
    module_names["config.params"] = os.path.join(ROOT, "config", "params.py")
    internal = set(module_names)


def parse_imports(path):
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as fh:
        tree = ast.parse(fh.read(), filename=path)
    deps = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                deps.add(alias.name)  # keep the full dotted name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                deps.add(node.module)
    return deps


# Resolve an imported name to the longest existing first-party module.
def resolve(name: str):
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        cand = ".".join(parts[:i])
        if cand in internal:
            return cand
    return None


edges = {}          # module -> set(first-party targets)
for mod, path in module_names.items():
    targets = set()
    for dep in parse_imports(path):
        resolved = resolve(dep)
        if resolved is not None:
            targets.add(resolved)
    edges[mod] = targets

print("== modules:", len(module_names))
print("== internal import edges:", sum(len(v) for v in edges.values()))

# --- cycles (SCCs of size>1) ---------------------------------------------
import collections

g = {m: {t for t in ts if t in internal} for m, ts in edges.items()}
rev = collections.defaultdict(set)
for m, ts in g.items():
    for t in ts:
        rev[t].add(m)

# Tarjan SCC
index = {}
lowlink = {}
on_stack = set()
stack = []
sccs = []
counter = [0]


def strongconnect(v):
    index[v] = lowlink[v] = counter[0]
    counter[0] += 1
    stack.append(v)
    on_stack.add(v)
    for w in g.get(v, ()):
        if w not in index:
            strongconnect(w)
            lowlink[v] = min(lowlink[v], lowlink[w])
        elif w in on_stack:
            lowlink[v] = min(lowlink[v], index[w])
    if lowlink[v] == index[v]:
        comp = []
        while True:
            w = stack.pop()
            on_stack.remove(w)
            comp.append(w)
            if w == v:
                break
        if len(comp) > 1:
            sccs.append(sorted(comp))


for v in g:
    if v not in index:
        strongconnect(v)

print("\n== import cycles (SCC size>1):", len(sccs))
for comp in sorted(sccs, key=len, reverse=True):
    print("  cycle[%d]: %s" % (len(comp), ", ".join(comp)))

# --- deployment spine ------------------------------------------------------
SEED = {"uav_isac.environment.env_core", "uav_isac.agents.trainer"}
closure = set(SEED)
frontier = set(SEED)
while frontier:
    nxt = set()
    for m in frontier:
        for t in g.get(m, ()) & internal:
            if t not in closure:
                closure.add(t)
                nxt.add(t)
    frontier = nxt

outside = sorted(m for m in internal if m not in closure)
print("\n== deployment spine size (env_core+trainer transitive):", len(closure))
print("== modules OUTSIDE the spine (research/tool candidates):", len(outside))
for m in outside:
    print("   ", m)

# --- modules with zero first-party importers -------------------------------
importers = collections.defaultdict(set)
for m, ts in g.items():
    for t in ts:
        importers[t].add(m)
entry_only = sorted(m for m in internal
                    if not importers[m] and m not in SEED)
print("\n== modules imported by no uav_isac module (entry/leaf):", len(entry_only))
for m in entry_only:
    print("   ", m)