import os, re

root = 'D:/BYLW/LD3'
cfg_dir = os.path.join(root, 'config')
existing = set(f for f in os.listdir(cfg_dir) if os.path.isfile(os.path.join(cfg_dir, f)))
refs = set()
search_dirs = ['docs', 'tools', 'scripts', 'tests', 'paper', 'uav_isac', 'config']
for sd in search_dirs:
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
            for m in re.finditer(r'config/([A-Za-z0-9_.\-]+\.(?:yaml|yml|json|py))', txt):
                refs.add(m.group(1))
            for m in re.finditer(r'"([A-Za-z0-9_]+\.(?:yaml|yml|json))"', txt):
                refs.add(m.group(1))
            for m in re.finditer(r"'([A-Za-z0-9_]+\.(?:yaml|yml|json))'", txt):
                refs.add(m.group(1))
            for m in re.finditer(r'extends:\s*([A-Za-z0-9_.\-]+)', txt):
                refs.add(m.group(1) + ('.yaml' if '.' not in m.group(1) else ''))
            # concatenated string fragments ending in .yaml
            for m in re.finditer(r'"([A-Za-z0-9_]+\.yaml)"', txt):
                refs.add(m.group(1))
            for m in re.finditer(r"'([A-Za-z0-9_]+\.yaml)'", txt):
                refs.add(m.group(1))

orphans = sorted(f for f in existing if f not in refs)
print('total config files:', len(existing), 'orphans:', len(orphans))
for f in orphans:
    print(' ', f)
