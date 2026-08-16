import os, re, json

root = 'D:/BYLW/LD3'
results_dir = os.path.join(root, 'results')
archive_dir = os.path.join(results_dir, '_archive')
live_dirs = set(d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)))
archived = set(d for d in os.listdir(archive_dir) if os.path.isdir(os.path.join(archive_dir, d)))

# Collect every reference to a results/ dir name anywhere in the tracked tree
refs = set()
search_dirs = ['docs', 'tools', 'scripts', 'tests', 'paper', 'uav_isac']
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
            for m in re.finditer(r'results/([A-Za-z0-9_.\-]+)', txt):
                name = m.group(1).rstrip('/')
                if '/' in name:
                    name = name.split('/')[0]
                if name and name not in ('.', '...'):
                    refs.add(name)

# Bare quoted dir names in tools (table-based tools reference without results/ prefix)
for fn in ['summarize_final_paper_suite.py', 'summarize_paper_baselines.py',
           'summarize_algorithm_baselines.py', 'summarize_algorithm_seed_stability.py',
           'summarize_ippo_seed_stability.py', 'summarize_training_seed_stability.py']:
    p = os.path.join(root, 'tools', fn)
    if os.path.isfile(p):
        txt = open(p, encoding='utf-8', errors='ignore').read()
        for m in re.finditer(r'"([a-z][a-z0-9_]*_[a-z0-9_]+)"', txt):
            refs.add(m.group(1))
        for m in re.finditer(r"'([a-z][a-z0-9_]*_[a-z0-9_]+)'", txt):
            refs.add(m.group(1))

# Tool default output dirs (referenced as defaults, not created yet)
tool_defaults = {
    'paper_final_suite', 'paper_algorithm_baselines', 'paper_algorithm_seed_stability',
    'paper_ippo_training_stability', 'token_semantic_probe', 'token_semantic_decoder_pretrain',
    'causal_token_value_probe', 'gate1a_llr_validation', 'gate1b_quantized_evidence',
    'isac_physical_feasibility_oracle', 'qos_commitment_dagger', 'qos_commitment_pretrain',
    'dagger_variants', 'dagger_corrected', 'full_eh', 'tica',
}
refs |= tool_defaults

print('=== ARCHIVED dirs still referenced by current tree (potential dead links):')
hits = []
for d in sorted(archived):
    if d in refs:
        hits.append(d)
print(len(hits))
for d in hits:
    print('  ', d)

print()
print('=== LIVE results dirs referenced by current tree:', len(live_dirs & refs))

# also check _archive prefix references
print()
print('=== does anything reference _archive itself?')
arch_refs = [r for r in refs if r.startswith('_archive')]
print(arch_refs if arch_refs else 'none')
