#!/usr/bin/env python3
"""Figure for the review experiments: paper/figures/eab_frontier.pdf -- total cost relative to IVF measured in the same
session (maintenance + Q x per-query time at recall 0.95 over the stream) against queries per update, one panel per
stream; methods that miss recall 0.95 in a measured round are omitted and named in the panel. Reads
results/bench_r_summary.json (bench/analyze_review.py)."""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 7.5, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
import os; F = json.load(open(os.environ.get('RSUMMARY', 'results/bench_r_summary.json')))['FRONTIER']['0.95']
NAME = {'ivf': 'IVF', 'ivf_part': 'label-part. IVF', 'scan_part': 'exact scan', 'acorn': 'ACORN', 'acorn_repair_all': 'ACORN + repair',
        'postfilter': 'post-filter', 'curator': 'Curator', 'rig': 'RIG', 'fdiskann_rebuild': 'F-DiskANN rebuild', 'ema': 'EMA', 'ivf_rt5': 'IVF, retrained'}
STY = {'ivf': ('C0', '-'), 'ivf_part': ('C3', '-'), 'ivf_rt5': ('C0', '--'), 'scan_part': ('0.5', '--'), 'acorn': ('C1', '-'), 'acorn_repair_all': ('C1', ':'),
       'postfilter': ('C2', '-'), 'curator': ('C4', '-'), 'rig': ('C7', '-'), 'fdiskann_rebuild': ('C5', '-'), 'ema': ('C6', '-'), 'policy': ('C5', '--')}
TITLE = {'sift500k': 'SIFT-500K (uniform)', 'sift500k_adv': 'SIFT-500K-adv (correlated)', 'deep5m': 'DEEP-5M (uniform)', 'arxiv_time': 'arXiv, time order (4096-d)'}
SN = [nm for nm in ('sift500k', 'sift500k_adv', 'deep5m', 'arxiv_time') if nm in F]
fig, axs = plt.subplots(1, len(SN), figsize=(7.1, 2.2), sharey=True); handles = {}
for ax, nm in zip(axs, SN):
    T = F[nm]['relative_to_ivf']
    pol = {m: v for m, v in T.items() if m.startswith('fdiskann_policy')}
    if pol:   # the executed rebuild policies: the cheapest feasible period at each query rate, as one line
        keys = set().union(*pol.values()); T = {m: v for m, v in T.items() if m not in pol}
        T['policy'] = {k: min(v[k] for v in pol.values() if k in v) for k in keys}
    for m, v in T.items():
        q = sorted(v, key=float); x = [float(k) for k in q]; y = [v[k] for k in q]; c, ls = STY.get(m, ('k', '-'))
        h, = ax.plot(x, y, color=c, ls=ls, lw=1.1, marker='o', ms=2); handles.setdefault(m, h)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(0.05, 300); ax.axhline(1, color='0.7', lw=0.6); ax.set_title(TITLE[nm], fontsize=7.5)
    ax.set_xlabel('queries per update'); inf = F[nm]['infeasible']
    ax.set_xticks([0.01, 1, 100]); ax.set_xticklabels(['0.01', '1', '100']); ax.minorticks_off()
    if inf:
        nn = [NAME.get(m, m) for m in sorted(inf)]
        ax.text(0.98, 0.98, 'misses 0.95:\n' + '\n'.join(', '.join(nn[i:i + 2]) for i in range(0, len(nn), 2)), transform=ax.transAxes, fontsize=5.8, color='0.3', va='top', ha='right')
axs[0].set_ylabel('total cost / IVF (same session)')
NAME['policy'] = 'F-DiskANN + delta, best period'
fig.legend([handles[m] for m in handles], [NAME.get(m, m) for m in handles], loc='upper center', ncol=6, frameon=False, fontsize=6.2, bbox_to_anchor=(0.5, 1.10))
fig.tight_layout(); fig.savefig('paper/figures/eab_frontier.pdf', bbox_inches='tight'); print('frontier figure written')
