import os, re

root = 'D:/BYLW/LD3'
results_dir = os.path.join(root, 'results')
archive_dir = os.path.join(results_dir, '_archive')
archived = set(d for d in os.listdir(archive_dir) if os.path.isdir(os.path.join(archive_dir, d)))

# Per-source-dir references: only live (non-archive) files
live_refs = {}
for sd in ['docs', 'tools', 'scripts', 'tests', 'uav_isac']:
    base = os.path.join(root, sd)
    if not os.path.isdir(base):
        continue
    for dirpath, _, files in os.walk(base):
        for fn in files:
            if not fn.endswith(('.md', '.py', '.yaml', '.yml', '.json')):
                continue
            p = os.path.join(dirpath, fn)
            try:
                txt = open(p, encoding='utf-8', errors='ignore').read()
            except Exception:
                continue
            for m in re.finditer(r'results/([A-Za-z0-9_.\-]+)', txt):
                name = m.group(1).rstrip('/')
                if '/' in name:
                    name = name.split('/')[0]
                if name and name not in ('.', '...'):
                    live_refs.setdefault(name, []).append(os.path.relpath(p, root))
            # bare quoted dir names that look like experiment dirs in tools
            if sd == 'tools':
                for m in re.finditer(r'"([a-z][a-z0-9_]*_[a-z0-9_]+)"', txt):
                    live_refs.setdefault(m.group(1), []).append(os.path.relpath(p, root) + ' (bare)')
                for m in re.finditer(r"'([a-z][a-z0-9_]*_[a-z0-9_]+)'", txt):
                    live_refs.setdefault(m.group(1), []).append(os.path.relpath(p, root) + ' (bare)')

print('=== ARCHIVED dirs referenced by LIVE files (dead links created):')
count = 0
for d in sorted(archived):
    if d in live_refs:
        count += 1
        print(f'  {d}')
        for src in sorted(set(live_refs[d]))[:4]:
            print(f'      <- {src}')
print('count:', count)

# bare-name-only additions from tools (these are tool tables)
tool_bare = set()
for sd in ['tools']:
    base = os.path.join(root, sd)
    for dirpath, _, files in os.walk(base):
        for fn in files:
            if not fn.endswith('.py'):
                continue
            p = os.path.join(dirpath, fn)
            try:
                txt = open(p, encoding='utf-8', errors='ignore').read()
            except Exception:
                continue
            for m in re.finditer(r'"([a-z][a-z0-9_]*_[a-z0-9_]+)"', txt):
                tool_bare.add(m.group(1))
print()
print('=== archived dirs matched ONLY as bare names in tools (table refs):')
for d in sorted(archived):
    if d in tool_bare:
        print('  ', d)
