import os, re

root = 'D:/BYLW/LD3'
results_dir = os.path.join(root, 'results')
existing = set(d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)))
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
            for m in re.finditer(r'"([a-z][a-z0-9_]*_[a-z0-9_]+)"', txt):
                refs.add(m.group(1))
            for m in re.finditer(r"'([a-z][a-z0-9_]*_[a-z0-9_]+)'", txt):
                refs.add(m.group(1))

# tool table names (bare dir names in summarize_* tools)
extra = {
    'paper_top1_test100', 'paper_top4_test100', 'paper_top1_stress50',
    'paper_top1_snr35_test50', 'paper_top1_deadline1ms_test50',
    'paper_top1_deadline0p8ms_test50', 'paper_no_movement_consensus_test100',
    'paper_fixed_split25_test100', 'distributed_consensus_hybrid50_bid_frozen_test100',
    'distributed_consensus_bid_token_frozen_test100',
    'distributed_consensus_intrinsic_bid_frozen_test100', 'baseline_silence_test100',
    'baseline_zero_payload_test100', 'baseline_permute_identity_test100',
    'baseline_no_comm_sensing_test100', 'baseline_no_capacity_test100',
    'baseline_central_movement_oracle_test100', 'qos_pretrained_soft_gated_move50_test100',
    'paper_top1_training_seed42_formal_test100', 'paper_top1_training_seed123_formal_test100',
    'paper_top1_training_seed456_formal_test100', 'paper_training_seed42_formal',
    'paper_training_seed123_formal', 'paper_training_seed456_formal',
    'paper_top1_training_seed42_test100', 'paper_top1_training_seed123_test100',
    'paper_top1_training_seed456_test100', 'paper_mappo_top1_seed42_critic_aligned_train40_test100',
    'paper_mappo_top1_seed123_critic_aligned_train40_test100',
    'paper_mappo_top1_seed456_critic_aligned_train40_test100',
    'paper_ippo_top1_seed42_critic_aligned_train40_test100',
    'paper_ippo_top1_seed123_critic_aligned_train40_test100',
    'paper_ippo_top1_seed456_critic_aligned_train40_test100',
    'paper_mappo_top1_seed42_train40_test100', 'paper_ippo_top1_seed42_train40_test100',
    'paper_ippo_training_stability', 'paper_algorithm_baselines',
    'paper_algorithm_seed_stability', 'paper_final_suite',
    'dagger_variants', 'dagger_corrected', 'full_eh', 'tica', 's4',
    'architecture_v2_scale_k6q6_local_move_rank_train10',
    'architecture_v2_scale_k6q6_local_move_ranker_gate_c1_6',
    'architecture_v2_scale_k6q6_teacher_trace_selection20',
    'architecture_v2_scale_k6q6_local_candidate_value_lu4_lq4_r1_gate10',
    'architecture_v2_scale_k6q6_local_candidate_value_lu4_lq4_r1_gate_a2_holdout10',
    'architecture_v2_scale_k6q6_factor_graph_gate_c1_screen',
    'architecture_v2_scale_k6q6_replicated_consensus_holdout10',
    'architecture_v2_teacher_cleanreset_trace_gate100',
    'architecture_v2_scale_k8q8_teacher_trace_d079_test20',
    'token_semantic_probe', 'token_semantic_decoder_pretrain', 'causal_token_value_probe',
    'gate1a_llr_validation', 'gate1b_quantized_evidence', 'isac_physical_feasibility_oracle',
    'qos_commitment_dagger', 'qos_commitment_pretrain',
}
refs |= extra

orphans = sorted(d for d in existing if d not in refs)
print('total existing dirs:', len(existing))
print('referenced:', len(existing) - len(orphans))
print('=== ORPHAN (never referenced by name in current tree):', len(orphans))
with open('D:/BYLW/LD3/_orphan_v3.txt', 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(orphans))
