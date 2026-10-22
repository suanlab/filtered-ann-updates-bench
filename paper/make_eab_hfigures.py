#!/usr/bin/env python3
"""Figures for the follow-up experiments (main env): paper/figures/eab_law.pdf (H1: crossover admissible-set size
vs n per method, with the sqrt(n) bound's slope for reference), paper/figures/eab_teaser.pdf (Figure 1: the same at recall
0.95 with arXiv, against fixed thresholds) and paper/figures/eab_percolation.pdf (H3: per-label
maximum recall against the effective admissible degree; maximum recall by round with and without repair).
Reads results/bench_h_summary.json and the raw H3 stream results."""
import json, glob, os, sys
sys.path.insert(0, 'bench')
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150, 'pdf.fonttype': 42, 'ps.fonttype': 42})
S = json.load(open('results/bench_h_summary.json'))
LAB = {'acorn': 'tombstone', 'acorn_repair': 'level-0 repair', 'acorn_repair_all': 'all-level repair'}
STY = {'ivf': ('IVF-Flat + bitmap', 'C0', 'o'), 'acorn': ('ACORN-$\\gamma$', 'C1', 's'), 'postfilter': ('HNSW post-filter', 'C2', '^'),
       'curator': ('Curator', 'C4', 'D'), 'ivf_part': ('label-partitioned IVF', 'C3', 'v'), 'fdiskann': ('Filtered-DiskANN', 'C5', 'P')}
NS = {'deep300k': 316_228, 'deep1m': 1_000_000, 'deep3m': 3_162_278, 'deep10m': 10_000_000}

def law():
    H = S.get('H1')
    if not H: return
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, t in zip(axs, ('0.95', '0.99')):
        rows = H['per_n'].get(t, {})
        for m, (lab, c, mk) in STY.items():
            xs, ys, cen = [], [], []
            for ds, r in rows.items():
                if m in r: xs.append(NS[ds]); ys.append(r[m]['m_star']); cen.append(r[m]['censor'])
            if not xs: continue
            o = np.argsort(xs); xs, ys, cen = np.array(xs)[o], np.array(ys)[o], np.array(cen)[o]
            ok = cen == 'interp'
            ax.plot(xs[ok], ys[ok], marker=mk, color=c, lw=1.2, ms=4, label=lab)
            ax.plot(xs[~ok], ys[~ok], marker=mk, color=c, lw=0, ms=4, mfc='none')
            ex = H['fit'].get(t, {}).get(m)
            if ex: ax.annotate(f"$n^{{{ex['exponent']:.2f}}}$", (xs[ok][-1], ys[ok][-1]), xytext=(4, 0), textcoords='offset points', color=c, fontsize=7, va='center')
        x = np.array([3e5, 1e7]); ax.plot(x, 3e4 * np.sqrt(x / 1e6), color='0.6', ls=':', lw=1); ax.text(1.2e7, 3e4 * np.sqrt(10), '$\\sqrt{n}$', color='0.4', fontsize=7)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('n (DEEP subsets, $\\beta{=}0$)'); ax.set_title(f'recall $\\geq${t}', fontsize=8)
    axs[0].set_ylabel('crossover $|A|$ (index = scan)')
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=6, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(); fig.savefig('paper/figures/eab_law.pdf', bbox_inches='tight'); plt.close(fig)

NS_AX = {'arxiv100k': 100_000, 'arxiv300k': 316_228, 'arxiv1m': 999_424}

def teaser():
    """Figure 1: crossover m* vs n at recall 0.95 (DEEP, beta=0; IVF also on 4096-d arXiv) against the two ways a
    dispatcher threshold is usually stated -- a fixed admissible-set size (horizontal) and a fixed selectivity (slope 1)."""
    H = S.get('H1'); AX = (S.get('AX') or {}).get('H1')
    if not H: return
    fig, ax = plt.subplots(figsize=(3.45, 2.35)); t = '0.95'
    def series(rows, ns, m):
        pts = sorted((ns[ds], r[m]['m_star'], r[m]['censor']) for ds, r in rows.items() if m in r and ds in ns)
        return np.array([p[0] for p in pts]), np.array([p[1] for p in pts]), np.array([p[2] for p in pts])
    for m in ('ivf', 'acorn', 'postfilter', 'curator', 'ivf_part'):
        lab, c, mk = STY[m]; xs, ys, cen = series(H['per_n'][t], NS, m); ok = cen == 'interp'
        ax.plot(xs[ok], ys[ok], marker=mk, color=c, lw=1.2, ms=3.5, label=lab); ax.plot(xs[~ok], ys[~ok], marker=mk, color=c, lw=0, ms=3.5, mfc='none')
        ex = H['fit'][t].get(m); dy = {'acorn': 3, 'postfilter': -3}.get(m, 0)
        if ex: ax.annotate(f"$n^{{{ex['exponent']:.2f}}}$", (xs[ok][-1], ys[ok][-1]), xytext=(3, dy), textcoords='offset points', color=c, fontsize=6.5, va='center')
    if AX:
        xs, ys, cen = series(AX['per_n'][t], NS_AX, 'ivf'); ok = cen == 'interp'
        ax.plot(xs[ok], ys[ok], marker='o', color='C0', lw=1.0, ls='--', ms=3.5, mfc='white', label='IVF, arXiv 4096-d')
        ex = AX['fit'][t].get('ivf')
        if ex: ax.annotate(f"$n^{{{ex['exponent']:.2f}}}$", (xs[ok][-1], ys[ok][-1]), xytext=(3, 0), textcoords='offset points', color='C0', fontsize=6.5, va='center')
    x = np.array([8e4, 1.6e7])
    ax.plot(x, [1e4, 1e4], color='0.45', lw=0.8, ls=(0, (4, 2)), label='fixed $|A|{=}10^4$')
    ax.plot(x, 1e-2 * x, color='0.45', lw=0.8, ls=(0, (1, 1.5)), label='fixed $s{=}10^{-2}$')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(8e4, 1.6e7); ax.set_ylim(7e2, 4e6)
    ax.set_xlabel('collection size $n$ (labels independent of geometry)'); ax.set_ylabel('crossover $|A|$ at recall 0.95')
    ax.legend(loc='upper left', fontsize=5.6, frameon=False, ncol=3, handlelength=1.8, columnspacing=0.7, borderaxespad=0.2)
    fig.tight_layout(); fig.savefig('paper/figures/eab_teaser.pdf', bbox_inches='tight'); plt.close(fig)

def percolation():
    from analyze import passes
    rows = []
    for root in passes('h3') + passes('h3b'):
        for f in glob.glob(f'{root}/stream_*/acorn*.json'):
            d = json.load(open(f)); nm = f.split('/')[-2][len('stream_'):]
            for r in d['rounds']:
                if 'perc' not in r or 'sweep' not in r: continue
                for lab in r['perc']:
                    b = str(lab['label']); rec = max(s['recall_by_label'].get(b, np.nan) for s in r['sweep'])
                    rows.append((nm, d['method'], r['round'], lab['deg_mean'], lab['lcc_frac'], rec))
    if not rows: return
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for nm, meth, c, mk in (('sift500k', 'acorn', 'C0', 'o'), ('sift500k_adv', 'acorn', 'C3', 's'), ('sift500k_adv', 'acorn_repair', 'C2', '^'), ('sift500k_adv', 'acorn_repair_all', 'C4', 'D')):
        R = [x for x in rows if x[0] == nm and x[1] == meth]
        if R: axs[0].scatter([x[3] for x in R], [x[5] for x in R], s=10, color=c, marker=mk, alpha=0.7, label=f"{nm.replace('sift500k_adv', 'SIFT-500K-adv').replace('sift500k', 'SIFT-500K')} {LAB[meth]}")
    axs[0].set_xscale('symlog', linthresh=1); axs[0].set_xlabel('mean effective admissible degree (label, round)'); axs[0].set_ylabel('max recall of the label')
    axs[0].axvline(1, color='0.6', ls=':', lw=1); axs[0].legend(frameon=False, fontsize=6.5)
    S3 = S.get('H3', {}).get('streams', {})
    for nm, ls in (('sift500k_adv', '-'), ('sift500k', '--')):
        for meth, c in (('acorn', 'C3'), ('acorn_repair', 'C2'), ('acorn_repair_all', 'C4')):
            d = S3.get(nm, {}).get(meth)
            if d: axs[1].plot([r[0] for r in d['rounds']], [r[1] for r in d['rounds']], ls=ls, color=c, marker='o', ms=3,
                              label=f"{nm.replace('sift500k_adv', 'SIFT-500K-adv').replace('sift500k', 'SIFT-500K')} {LAB[meth]}")
    axs[1].set_xlabel('round'); axs[1].set_ylabel('max recall'); axs[1].legend(frameon=False, fontsize=6.5)
    fig.tight_layout(); fig.savefig('paper/figures/eab_percolation.pdf', bbox_inches='tight'); plt.close(fig)

law(); teaser(); percolation(); print('follow-up figures written')
