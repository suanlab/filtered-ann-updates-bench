#!/usr/bin/env python3
"""Figure of follow-up 2's mechanism runs (tag r8): paper/figures/eab_reach.pdf. One point per label and measured round:
the share of the label's live nodes reachable from its queries' level-0 entry (ACORN's filtered descent replayed exactly,
then the effective admissible edges) against the label's maximum recall, per update path, on SIFT-500K-adv (correlated
deletions) and DEEP-5M (uniform). Reads results/bench_r8 (bench/stream_acorn.py with MECH=1)."""
import json, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 7.5, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
MODES = (('acorn', 'tombstones', 'C3', 'o'), ('acorn_repair', 'level-0 re-rank', 'C2', '^'), ('acorn_repair_upper', 'upper levels only', 'C1', 'v'),
         ('acorn_repair_all', 'all-level re-rank', 'C4', 'D'), ('acorn_repair_fill', 'slot-fill', 'C0', 's'))
TITLE = {'sift500k_adv': 'SIFT-500K-adv (correlated deletions)', 'deep5m': 'DEEP-5M (uniform deletions)'}
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.35)); handles = {}
for ax, nm in zip(axs, ('sift500k_adv', 'deep5m')):
    for m, lab, c, mk in MODES:
        f = f'results/bench_r8/stream_{nm}/{m}.json'
        if not os.path.exists(f): continue
        d = json.load(open(f)); xs, ys = [], []
        for r in d['rounds']:
            if 'mech' not in r or 'sweep' not in r: continue
            for L in r['mech']['labels']:
                b = str(L['label']); rec = max(s['recall_by_label'].get(b, np.nan) for s in r['sweep'])
                if np.isfinite(rec): xs.append(L['reach_frac']); ys.append(rec)
        h = ax.scatter(xs, ys, s=9, color=c, marker=mk, alpha=0.75, linewidths=0); handles.setdefault(lab, h)
    ax.plot([0, 1], [0, 1], color='0.75', lw=0.6, ls=':'); ax.set_title(TITLE[nm], fontsize=7.5)
    ax.set_xlabel("label's live nodes reachable from the entry"); ax.set_ylabel('max recall of the label')
axs[0].set_xlim(0, 1.02); axs[0].set_ylim(0, 1.02); axs[1].set_xlim(0.8, 1.005); axs[1].set_ylim(0.8, 1.005); axs[1].set_xticks([0.8, 0.85, 0.9, 0.95, 1.0])
fig.legend(list(handles.values()), list(handles.keys()), loc='upper center', ncol=5, frameon=False, fontsize=6.4, bbox_to_anchor=(0.5, 1.07))
fig.tight_layout(); fig.savefig('paper/figures/eab_reach.pdf', bbox_inches='tight'); print('reach figure written')
