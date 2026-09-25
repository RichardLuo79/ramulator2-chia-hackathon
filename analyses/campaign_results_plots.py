"""Historical exploratory plot helpers, retained for evidence regression tests.

The current paper notebook uses paper_plots.py and paper_workflow.py. These
helpers consume the original comprehensive evidence; none accesses cloud
storage, runs a simulator, or imports the campaign harness.
"""
from pathlib import Path
import base64
import html
import io
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from IPython.display import display, HTML, Markdown

LABELS = {'astra_single_core': 'Astra · SC', 'astra_multicore': 'Astra · MC',
          'deepseek_single_core': 'DeepSeek · SC', 'deepseek_multicore': 'DeepSeek · MC',
          'opus_single_core': 'Opus 5.5 · SC', 'gemini_single_core': 'Gemini · SC',
          'fixedlat': 'Queued FixedLat', 'md1': 'M/D/1', 'wmg1': 'WMG1', 'mess': 'MeSS',
          'oracle': 'Cycle-level oracle'}
COLORS = {'astra_single_core': '#0072B2', 'astra_multicore': '#0072B2',
          'deepseek_single_core': '#D55E00', 'deepseek_multicore': '#D55E00',
          'opus_single_core': '#E69F00', 'gemini_single_core': '#56B4E9',
          'fixedlat': '#777777', 'md1': '#CC79A7', 'wmg1': '#009E73', 'mess': '#6B58A5',
          'oracle': '#222222'}
SYNTH = ['astra_single_core', 'astra_multicore', 'deepseek_single_core', 'deepseek_multicore']
BASELINES = ['fixedlat', 'md1', 'wmg1', 'mess']
# Extension/diagnostic matrices remain the original frozen four endpoints.
# Availability in one study must not silently add an arm to another study.
FOUNDATION_SYNTH = ['astra_single_core', 'deepseek_single_core', 'opus_single_core', 'gemini_single_core']
FOUNDATION = BASELINES + FOUNDATION_SYNTH
LAT_TP_FOUNDATION = BASELINES + ['astra_single_core', 'deepseek_single_core']
ALL_ARMS = BASELINES + SYNTH
SPLITS = ['training', 'validation', 'test']
PRETTY = {'core_abs_pct': 'Core error (%)', 'core_signed_pct': 'Signed core error (%)',
          'mae_L': 'Request MAE/L', 'drift_L': 'Signed drift/L', 'abs_drift_L': 'Absolute drift/L',
          'p99_L': 'P99 |error|/L', 'p999_L': 'P99.9 |error|/L', 'worst_core_pct': 'Worst core (%)',
          'makespan_abs_pct': 'Makespan error (%)', 'oracle_coverage': 'Oracle pairing (%)',
          'model_coverage': 'Model pairing (%)', 'pairs': 'Paired reads', 'L': 'Oracle L (cycles)',
          'case': 'Workload / mix', 'arm': 'Model', 'cohort': 'Split', 'cores': 'Cores',
          'cases': 'Cases', 'min_cycles': 'Min error (cycles)', 'max_cycles': 'Max error (cycles)'}


def setup(data, output='figures/chia_campaign_results'):
    global FIGURES, TABLES, DATA, DF
    DATA = data
    DF = pd.DataFrame(data['rows'])
    FIGURES = Path(output); TABLES = FIGURES.parent.parent / 'tables/chia_campaign_results'
    FIGURES.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.labelsize': 9, 'axes.titlesize': 10, 'legend.fontsize': 8,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .7, 'lines.linewidth': 1.6,
        'figure.dpi': 130, 'savefig.dpi': 180, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'axes.unicode_minus': True})


def finish(fig, name):
    # Solve layout once on the notebook renderer. Re-solving on each PDF/SVG/PNG
    # backend can shift axis-label positions between bounding-box calculations.
    fig.canvas.draw()
    fig.set_layout_engine('none')
    # Axis label positions otherwise get recomputed from backend-specific tick
    # extents during savefig's tight-bbox pass (which can clip a left label).
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            label = axis.label
            position = fig.transFigure.inverted().transform(
                label.get_transform().transform(label.get_position()))
            axis.set_label_coords(*position, transform=fig.transFigure)
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(FIGURES / f'{name}.{suffix}', bbox_inches='tight', pad_inches=.08)
    plt.show(); plt.close(fig)


def opus_companion():
    frame = pd.DataFrame(DATA['opus']['rows'])
    assert frame.groupby('cohort').size().to_dict() == {'test':24,'training':14,'validation':14}
    headline = frame.groupby('cohort',sort=False).agg(
        cases=('case','size'),core_abs_pct=('core_abs_pct','mean'),mae_L=('mae_L','mean'),
        abs_drift_L=('abs_drift_L','mean'),p999_L=('p999_L','mean'),
        oracle_coverage=('oracle_coverage','mean'),pairs=('pairs','sum')).reset_index()
    table(headline,'opus_single_core_headline',caption='Opus SC: equal-case means; no split pooling')
    table(frame.drop(columns=['populations','per_core_signed_pct']), 'opus_single_core_cases',
          caption='Opus SC: complete 14 / 14 / 24 case evidence',collapsed=True)


def standalone_speed(study_key='dram_speed', output_prefix='dram_speed'):
    """Render one host study at a time; never pool independent oracle timings."""
    study = DATA.get(study_key)
    if not study:
        display(Markdown('Standalone throughput results have not been collected.'))
        return
    frame = pd.DataFrame(study['summary'])
    arms = study['protocol']['models']
    assert len(arms) == len(set(arms)) and 'oracle' in arms
    assert set(frame.arm) == set(arms)
    assert len(frame) == 30 * len(arms) and not frame.duplicated(['arm','pattern','read_percent','interval']).any()
    assert set(frame.read_percent) == {100,75,50}
    assert frame.n.sum() == study['successful']
    display(Markdown(f"**Coverage:** {study['successful']:,}/{study['expected']:,} valid timings; "
        f"{study['failed']:,} failed and {study['pending']:,} pending. "
        "Results below are provisional until the five-repetition matrix finishes."
        if study['pending'] else
        f"**Coverage:** {study['successful']:,}/{study['expected']:,} valid timings; {study['failed']:,} failed."))
    successful = frame[frame.n > 0]
    shared = set.intersection(*[set(zip(g.pattern,g.read_percent,g.interval))
        for arm in arms for g in [successful[successful.arm==arm]]])
    headline=[]
    for arm in arms:
        g=successful[successful.arm==arm]
        same=g.loc[np.array([tuple(x) in shared for x in zip(g.pattern,g.read_percent,g.interval)],dtype=bool)]
        speed=same.oracle_relative_speedup.dropna()
        headline.append(dict(arm=arm,traffic_points=len(g),planned_points=30,
             min_repeats=int(g.n.min()) if len(g) else 0,max_repeats=int(g.n.max()) if len(g) else 0,
             shared_points=len(speed),geomean_speedup=float(np.exp(np.log(speed).mean())) if len(speed) else np.nan))
    table(pd.DataFrame(headline),output_prefix+'_coverage',caption='Coverage and speedup on shared traffic points within this host study')
    intervals=[1,4,16,64,256]
    for metric, name, ylabel, scale in (
            ('median',output_prefix+'_throughput','Million requests / host second',1e6),
            ('oracle_relative_speedup',output_prefix+'_speedup','Speedup over cycle-level oracle',1)):
        fig,axes=plt.subplots(2,3,figsize=(10.5,6.3),sharex=True,layout='constrained')
        positive=(pd.to_numeric(frame[metric],errors='coerce')/scale).dropna()
        positive=positive[positive>0]
        limits=(positive.min()/1.5,positive.max()*1.5) if len(positive) else (.1,10.)
        for row,pattern in enumerate(('streaming','random')):
            for col,ratio in enumerate((100,75,50)):
                ax=axes[row,col]
                for arm in arms:
                    g=frame[(frame.pattern==pattern)&(frame.read_percent==ratio)&(frame.arm==arm)].set_index('interval').reindex(intervals)
                    ax.plot(intervals,g[metric]/scale,color=COLORS[arm],label=LABELS[arm],
                        linestyle='--' if arm.endswith('multicore') else '-',
                        marker='s' if arm.endswith('multicore') else 'o',markersize=3,linewidth=1.35)
                ax.set_xlim(.8,320); ax.set_ylim(*limits)
                ax.set_xscale('log',base=4); ax.set_yscale('log')
                ax.set_xticks(intervals,[str(i) for i in intervals])
                if not ((frame.pattern==pattern)&(frame.read_percent==ratio)&(frame.n>0)).any():
                    ax.text(.5,.5,'Pending',ha='center',va='center',transform=ax.transAxes,color='#777777')
                ax.set_title(f'{pattern.capitalize()} · {ratio}% reads')
                ax.grid(axis='y',which='major',alpha=.22)
                if row==1:ax.set_xlabel('Offered interval (DRAM cycles)')
                if col==0:ax.set_ylabel(ylabel)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
        finish(fig,name)
    compact=frame.drop(columns='repetitions')
    table(compact,output_prefix+'_all_points',caption='Every traffic point and repeat count',collapsed=True)
    repeats=[]
    for row in study['summary']:
        for trial in row['repetitions']:
            repeats.append({k:row[k] for k in ('arm','pattern','read_percent','interval')}|trial)
    table(pd.DataFrame(repeats),output_prefix+'_repetitions',caption='Individual standalone timings',collapsed=True)
    line_note = ('Solid lines denote SC endpoints; dashed lines denote MC endpoints. '
                 if any(arm.endswith('multicore') for arm in arms) else
                 'The synthesized models are the frozen SC selections. ')
    pilot_note = ('The 100k/500k-request feasibility pilot is archived separately and is not plotted.'
                  if study_key == 'dram_speed' else
                  'Each speedup uses the fresh oracle from this study, not the earlier host.')
    display(Markdown(line_note + 'Missing points remain gaps. '
        'The primary timings do not include CSV serialization. ' + pilot_note))


def workflow():
    """Two-stage search and its information boundaries; no result data changes."""
    ink, muted, border = '#243746', '#536474', '#BDCAD3'
    blue, teal, purple = '#0072B2', '#007D73', '#76538E'
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 7.2),
                             gridspec_kw={'height_ratios': [1, 1.45]})
    fig.subplots_adjust(left=.025, right=.985, top=.985, bottom=.025, hspace=.10)
    for ax in axes:
        ax.set(xlim=(0, 100), ylim=(0, 100))
        ax.set_axis_off()

    def box(ax, x, y, w, h, title, body='', *, fill='#F7F9FB', edge=border):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle='round,pad=0,rounding_size=1.2',
            facecolor=fill, edgecolor=edge, linewidth=.8))
        if body:
            ax.text(x+w/2, y+h-4.2, title, ha='center', va='top',
                    fontsize=9.2, weight='bold', color=ink)
            ax.text(x+w/2, y+h-10.2, body, ha='center', va='top',
                    fontsize=8.3, linespacing=1.25, color=muted)
        else:
            ax.text(x+w/2, y+h/2, title, ha='center', va='center',
                    fontsize=9, linespacing=1.35, weight='bold', color=ink)

    def arrow(ax, start, end, *, color=muted):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
            mutation_scale=10, linewidth=1.0, color=color,
            shrinkA=2, shrinkB=2))

    ax = axes[0]
    ax.text(2, 97, '(a) Campaign schedule', fontsize=11, weight='bold', va='top', color=ink)
    ax.text(2, 86, 'Same simple seed; Astra / DeepSeek / Gemini: 10 + 5 rounds; Opus: 10 only', fontsize=8.8, color=muted)
    schedule = [
        (2, 13, 'Common\nseed', '#F7F9FB', border),
        (19, 21, 'Single-core search\nRounds 1–10', '#EEF6FB', blue),
        (44, 12, 'Retain\nSC endpoint', '#EAF5F1', teal),
        (60, 23, 'Multicore search\nRounds 11–15', '#EEF6FB', blue),
        (87, 11, 'Retain\nMC endpoint', '#EAF5F1', teal),
    ]
    for x, w, label, fill, edge in schedule:
        box(ax, x, 54, w, 24, label, fill=fill, edge=edge)
    for (x, w, *_), (nx, *_) in zip(schedule, schedule[1:]):
        arrow(ax, (x+w, 66), (nx, 66), color=blue)
    box(ax, 2, 5, 37, 31, 'Opus: stop after round 10',
        'Test the frozen SC endpoint\n24 single-core workloads; no MC stage',
        fill='#FFF6E4', edge='#E69F00')
    arrow(ax, (44, 54), (38, 36), color='#E69F00')
    box(ax, 43, 5, 55, 31, 'After round 15: test BOTH frozen endpoints',
        '24 singles + 8 four-core + 8 eight-core mixes\nTransfer studies: DRAM configurations and gem5',
        fill='#EAF5F1', edge=teal)
    arrow(ax, (50, 54), (50, 36), color=teal)
    arrow(ax, (92.5, 54), (92.5, 36), color=teal)
    ax.text(70, 43, 'No test or transfer feedback', ha='center', va='center',
            fontsize=8.4, color=teal)

    ax = axes[1]
    ax.text(2, 98, '(b) Inside each round', fontsize=11, weight='bold', va='top', color=ink)
    box(ax, 3, 65, 29, 25, 'Develop a candidate',
        'Inspect, edit, build; training tools\nOptional synthetic tests / replay', fill='#EEF6FB', edge=blue)
    box(ax, 36, 65, 28, 25, 'Freeze one submission',
        'No candidate edits after\nfinal validation or review')
    box(ax, 68, 65, 29, 25, 'Checks + evaluation',
        'Mechanical integrity is mandatory\nTraining + anonymous validation\nCore, request and tail metrics')
    box(ax, 68, 24, 29, 28, 'Independent LLM review',
        'Same model + effort\nCompare incumbent + stage entry\nJustified trade-offs allowed',
        fill='#F5F0F8', edge=purple)
    box(ax, 36, 24, 28, 28, 'Promote or retain',
        'Choose candidate or incumbent\nCommit the decision once\nNo same-round correction', fill='#EAF5F1', edge=teal)
    box(ax, 3, 24, 29, 28, 'Reflect and summarize',
        'Record evidence and lessons\nDo not edit the model', fill='#EEF6FB', edge=blue)
    arrow(ax, (32, 77.5), (36, 77.5))
    arrow(ax, (64, 77.5), (68, 77.5))
    arrow(ax, (82.5, 65), (82.5, 52), color=purple)
    arrow(ax, (68, 38), (64, 38), color=purple)
    arrow(ax, (36, 38), (32, 38), color=teal)
    arrow(ax, (7, 52), (7, 65), color=blue)
    ax.text(9, 58.5, 'Next round: selected model\n+ summary and own history',
            va='center', fontsize=8.1, color=blue, linespacing=1.2)
    ax.text(50, 12, 'Training: named inputs + diagnostics   |   Validation: anonymous numerical feedback only',
            ha='center', fontsize=8.7, color=muted)
    ax.text(50, 4, 'Tests stay outside the loop. An LLM cannot waive failed integrity checks.',
            ha='center', fontsize=8.7, color=muted)
    finish(fig, 'chia_loop_workflow')


def case_label(name):
    short = name.split('-c1-', 1)[-1] if '-c1-' in name else name
    replacements = {'ligra_cf': 'Ligra CF', 'ligra_radii': 'Ligra Radii', 'gap_bc': 'GAP BC',
        'biogpt': 'BioGPT', 'bark': 'Bark', 'cam4': 'CAM4', 'gcc': 'GCC', 'pop2': 'POP2',
        'xalancbmk': 'Xalancbmk', 'wrf': 'WRF', 'x264': 'x264'}
    if short in replacements: return replacements[short]
    if short.startswith('sierra_a_'):
        fields = short.split('_'); return f'Sierra {fields[2]} / {fields[3]}'
    if short.startswith('tahoe_'): return 'Tahoe / ' + short.split('_')[-1]
    if short.startswith('test-c4_'): return '4C ' + short[8:].replace('_', ' ')
    if short.startswith('test-c8-'): return '8C mix ' + short.split('-')[-1]
    return short.replace('_', ' ')


def table(frame, name, *, caption=None, collapsed=False, presentation=True):
    """Save full precision CSV and paper-ready LaTeX; never truncate the HTML."""
    frame.to_csv(TABLES / f'{name}.csv', index=False)
    shown = frame.copy()
    if presentation:
        if 'arm' in shown: shown['arm'] = shown.arm.map(LABELS)
        if 'case' in shown: shown['case'] = shown['case'].map(case_label)
        for col in ['oracle_coverage', 'model_coverage']:
            if col in shown: shown[col] = shown[col] * 100
        shown = shown.rename(columns=PRETTY)
    shown.to_latex(TABLES / f'{name}.tex', index=False, escape=True,
                   float_format=lambda x: f'{x:.4g}', na_rep='—')
    body = shown.to_html(index=False, float_format=lambda x: f'{x:.4g}', na_rep='—', border=0)
    body = '<div style="overflow-x:auto;max-width:100%">' + body + '</div>'
    if collapsed:
        body = f'<details><summary>{html.escape(caption or name)} ({len(frame)} rows)</summary>{body}</details>'
    elif caption:
        display(Markdown('**' + caption + '**'))
    display(HTML(body))


def campaign_usage_section():
    """Four ten-round campaigns; all accounting is in the embedded supplement."""
    study = DATA['single_core_usage']
    # The generator embeds campaign_usage.py immediately before this cell.
    validate(study, DATA['sources'])
    names = {'astra':'Astra', 'deepseek':'DeepSeek', 'gemini':'Gemini', 'opus':'Opus 5.5'}
    colors = {c:COLORS[c+'_single_core'] for c in names}
    h = pd.DataFrame(study['tables']['headline']).set_index('campaign').loc[list(names)]
    rounds = pd.DataFrame(study['tables']['rounds'])
    phases = pd.DataFrame(study['tables']['phases'])

    def cost_text(a, b, incomplete=False):
        if pd.isna(b):
            text = f'≥ ${a:,.2f}'
        elif b - a < .01:
            text = f'${a:,.2f}'
        else:
            text = f'${a:,.2f}–{b:,.2f}'
        return text + (' †' if incomplete else '')

    gaps = {
        'astra':'825 request records recovered; two interrupted tails have unknown additional usage.',
        'deepseek':'1,245 reported / 1,248 attempted requests; three lack usage; two reviews mechanically skipped.',
        'gemini':'All 2,017 recorded requests have input/output; 81 lack a cache partition.',
        'opus':'All 30 main-agent CLI invocations; not cumulative modelUsage or inferred subagent usage.'}
    display_rows = []
    for c, row in h.iterrows():
        display_rows.append({'Campaign':names[c]+(' †' if row.unknown_usage_records else ''),
            'Rounds': '10/10', 'Input (M)':row.input_tokens / 1e6,
            'Output (M)':row.output_tokens / 1e6, 'Cache read (M)':row.cached_input_tokens / 1e6,
            'Cache write (M)':row.cache_write_input_tokens / 1e6,
            'Reasoning (M; in output)':row.reasoning_output_tokens / 1e6,
            'API-equivalent cost':cost_text(row.observed_cost_lower_usd, row.observed_cost_upper_usd,
                                           row.unknown_usage_records > 0)})
    table(pd.DataFrame(display_rows), 'usage_headline', presentation=False,
          caption='Single-core campaign usage: recorded token volumes and estimated API-equivalent LLM cost')
    table(h.reset_index(), 'usage_headline_exact', presentation=False, collapsed=True,
          caption='Exact token counters, known-cost subtotals, and missing-counter counts')
    completeness = pd.DataFrame([{'Campaign':names[c], 'Invocations':study['campaigns'][c]['invocations'],
        'Coverage and gaps':gaps[c], 'Billing interpretation':study['campaigns'][c]['billing']}
        for c in names])
    table(completeness, 'usage_completeness', presentation=False, caption='Accounting completeness')

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.0), sharey=True, layout='constrained')
    y = np.arange(4)
    labels = [names[c]+(' †' if h.loc[c,'unknown_usage_records'] else '') for c in names]
    for ax, field, title in zip(axes[:2], ('input_tokens','output_tokens'),
                                ('Processed input', 'Output, including reasoning')):
        vals = h[field].to_numpy() / 1e6
        ax.barh(y, vals, color=[colors[c] for c in names], height=.56)
        for pos, value in zip(y, vals):
            ax.text(value + max(vals)*.02, pos, f'{value:.2f}', va='center', fontsize=8)
        ax.set_xlim(0, max(vals)*1.23); ax.set_xlabel('Million tokens')
        ax.set_title(title); ax.grid(axis='x', alpha=.14); ax.set_axisbelow(True)
    axes[0].set_yticks(y, labels); axes[0].invert_yaxis()
    ax = axes[2]
    for i, c in enumerate(names):
        r = h.loc[c]; a, b = r.observed_cost_lower_usd, r.observed_cost_upper_usd
        ax.hlines(i, a, b, color=colors[c], lw=4)
        ax.plot([a,b], [i,i], '|', color=colors[c], ms=9)
        if b-a < .01: ax.plot(a, i, 'o', color=colors[c], ms=5)
        txt = cost_text(a,b).replace('$','') + (' + ?' if r.unknown_usage_records else '')
        ax.annotate(txt, (b,i), xytext=(7,0), textcoords='offset points', va='center', fontsize=8)
    ax.set_xscale('log'); ax.set_xlim(4, 3000); ax.set_xticks([10,100,1000], ['$10','$100','$1,000'])
    ax.set_title('Estimated API-equivalent cost'); ax.set_xlabel('USD · logarithmic scale')
    ax.grid(axis='x', alpha=.14)
    finish(fig, 'single_core_token_cost')
    display(Markdown('† **Known subtotal only; additional usage is unknown.** Horizontal ranges reflect '
        'missing cache/tariff information, not measured variation. Values are rounded for readability; '
        'micro-dollar outward rounding is retained in the exported tables.'))

    fig, axes = plt.subplots(2, 4, figsize=(11.8, 5.0), layout='constrained')
    for col, c in enumerate(names):
        rows = rounds[rounds.campaign==c].sort_values('round')
        x = rows['round'].to_numpy()
        volume = (rows.input_tokens + rows.output_tokens).to_numpy()/1e6
        ax = axes[0,col]
        bars = ax.bar(x, volume, color=colors[c], alpha=.85)
        for bar, unknown in zip(bars, rows.unknown_usage_records):
            if unknown:
                bar.set_hatch('///'); bar.set_edgecolor('#444444')
        ax.set_title(names[c]); ax.set_ylim(0,max(volume)*1.12)
        ax.set_xticks([1,4,7,10]); ax.grid(axis='y',alpha=.14); ax.set_axisbelow(True)
        ax.set_ylabel('Input + output (M tokens)' if col==0 else '')
        ax = axes[1,col]
        lo = rows.cumulative_observed_cost_lower_usd.to_numpy()
        hi = rows.cumulative_observed_cost_upper_usd.to_numpy()
        unknown = rows.cumulative_unknown_usage_records.to_numpy() > 0
        ax.fill_between(x,lo,hi,color=colors[c],alpha=.20)
        for k in range(1,len(x)):
            style = '--' if unknown[k] else '-'
            ax.plot(x[k-1:k+1],lo[k-1:k+1],style,color=colors[c],lw=1.5)
            if hi[k]!=lo[k]: ax.plot(x[k-1:k+1],hi[k-1:k+1],style,color=colors[c],lw=1.5)
        for k in range(len(x)):
            ax.plot(x[k],lo[k],'o',color=colors[c],ms=3.5,
                    markerfacecolor='white' if unknown[k] else colors[c])
        ax.set_ylim(0,max(hi)*1.10); ax.set_xlabel('Round'); ax.set_xticks([1,4,7,10])
        ax.set_ylabel('Cumulative API-equivalent USD' if col==0 else '')
        ax.grid(axis='y',alpha=.14)
    finish(fig, 'single_core_usage_by_round')
    display(Markdown('Axes are scaled separately by campaign. Hatched bars mark incomplete round usage. '
        'Dashed cumulative segments and hollow markers remain **observed subtotals** after an unknown '
        'call; later completed rounds do not remove that gap. The Gemini band reflects unknown cache '
        'partitions. Input dominates processed volume; the tables retain output and reasoning separately.'))

    table(rounds, 'usage_by_round', presentation=False, collapsed=True,
          caption='Rounds 1–10: exact token subtotals, cost intervals, and persistent cumulative gaps')
    table(phases, 'usage_by_phase', presentation=False, collapsed=True,
          caption='Disjoint phases: development, review, reflection, and identifiable compaction')
    table(pd.DataFrame(study['tables']['attempt_status']), 'usage_successful_failed', presentation=False,
          collapsed=True, caption='Successful versus failed attempts (partial recorded usage retained)')
    failed = []
    for inv in study['invocations']:
        if inv['success']: continue
        parts = [r for r in study['rows'] if r['invocation_sha256']==inv['invocation_sha256']]
        s = summarize(parts)
        failed.append({'campaign':inv['campaign'], 'round':inv['round'], 'phase':inv['parent_phase'],
            'attempt':inv['attempt_number'], 'input_tokens':s['input_tokens'], 'output_tokens':s['output_tokens'],
            'observed_cost_lower_usd':s['observed_cost_lower_usd'],
            'observed_cost_upper_usd':s['observed_cost_upper_usd'], 'unknown_usage_records':s['unknown_usage_records'],
            'receipt_sha256':inv['receipt_sha256']})
    table(pd.DataFrame(failed), 'usage_failed_attempts', presentation=False, collapsed=True,
          caption='Failed-attempt ledger: failures do not erase recorded provider work')
    skips = [{'campaign':c, **s} for c in names for s in study['campaigns'][c]['skipped_reviews']]
    table(pd.DataFrame(skips), 'usage_skipped_reviews', presentation=False, collapsed=True,
          caption='Mechanically skipped reviews: no provider call and no missing-usage penalty')
    tariff_rows = []
    for c in names:
        t = study['campaigns'][c]['tariff']
        for i, tier in enumerate(t['tiers']):
            tariff_rows.append({'Campaign':names[c], 'Frozen date':t['checked_date'],
                'Request context': ('≤ '+str(tier['maximum_input_tokens'])) if tier['maximum_input_tokens'] else
                                   ('> '+str(t['tiers'][i-1]['maximum_input_tokens'])) if i else 'All',
                **{k:float(v) if v is not None else None for k,v in tier['rates'].items()},
                'Units':t['units'], 'Source':t['source'],
                'Tariff SHA256':study['campaigns'][c]['tariff_sha256']})
    table(pd.DataFrame(tariff_rows), 'usage_tariffs', presentation=False, collapsed=True,
          caption='Dated frozen tariffs: USD per million tokens, not current repricing')
    table(pd.DataFrame(study['invocations']), 'usage_invocations', presentation=False, collapsed=True,
          caption='Invocation reconciliation and source hashes (no prompts or native state)')
    # Export compact complete counters without rendering thousands of HTML rows.
    ledger = pd.json_normalize(study['rows'])
    ledger.to_csv(TABLES/'usage_accounting_ledger.csv',index=False)
    ledger.to_latex(TABLES/'usage_accounting_ledger.tex',index=False,escape=True,na_rep='—')
    display(Markdown('The full sanitized accounting ledger is exported as CSV/LaTeX. Its rows are '
        'bound to campaign, round, phase, attempt, receipt, and source hashes. CLI invocations and '
        'individual API requests are reported separately; they are not comparable call counts.'))
    display(Markdown('### What drove the recorded usage\n\n'
        '- **Development was the largest cost component** in all four campaigns. Astra round 7 '
        'and Opus round 8 were their most expensive recorded rounds; DeepSeek peaked in round 2. '
        'Gemini rounds 2 and 6 have the largest cost intervals, but their ordering is unresolved '
        'because those intervals overlap. Astra and DeepSeek rankings describe observed usage, not unknown tails.\n'
        '- **Cache reuse matters.** Reported cache reads account for about 37% of Astra input, '
        '95% of DeepSeek input, at least 94% of Gemini input, and 95% of Opus input. These are '
        'fractions of processed context, not fractions of unique text. Opus recorded 11.84M '
        'cache-write tokens, all with the one-hour duration; its estimate uses that rate.\n'
        '- **Compaction is not free, but it must be counted once.** DeepSeek has 32 explicitly '
        'identified compaction requests, about $1.74 of its recorded subtotal. They are separated '
        'from development/reflection above. Gemini has no separately identified compaction call '
        'in rounds 1–10. Native CLI compaction is not separately attributable and is not added '
        'on top of the authoritative usage totals.\n'
        '- **Recovery changes the accounting, not the experiment.** Astra\'s 825 checksummed '
        'request records replace cumulative CLI totals, recovering known failed-attempt work '
        'without charging earlier turns again. All recorded prompts are below the frozen '
        '272k long-context threshold (largest: 229,202 tokens). Two interrupted tails still '
        'have unknown usage. DeepSeek has three failed requests without counters; its round-1 '
        'recovery reuses the saved provider receipt once. Gemini\'s failed round-10 reflection '
        'has a recorded response, which is included along with its successful continuation.\n\n'
        'Opus\'s per-invocation raw counters also report a reasoning split omitted by the old '
        'normalizer; it is recovered here without increasing output or cost. All 30 invocation '
        'totals are available, but this is main-agent accounting, not a claim about unreported '
        'subagents. These are four individual campaigns with different tariffs, tokenizers, '
        'cache behavior, and development paths—not a general cost or intelligence ranking.'))
    display(Markdown('Tariff sources, as frozen at campaign launch: '
        '[Astra (2026-09-19)](https://developers.openai.com/api/docs/models/gpt-6-astra), '
        '[DeepSeek (2026-09-19)](https://api-docs.deepseek.com/quick_start/pricing/), '
        '[Gemini (2026-09-19)](https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing), '
        '[Opus (2026-09-23)](https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5). '
        'The embedded tariff records, rather than the current contents of these pages, determine the estimates.'))


def headline(frame, arms, groupby=('cohort', 'cores')):
    result = []
    for group, values in frame.groupby(list(groupby), sort=False):
        group = group if isinstance(group, tuple) else (group,)
        populations = [set(values.loc[values.arm == arm, 'case']) for arm in arms]
        shared = set.intersection(*populations) if populations else set()
        if not shared: continue
        for arm in arms:
            rows = values[(values.arm == arm) & values.case.isin(shared)]
            row = dict(zip(groupby, group), arm=arm, cases=len(rows))
            for metric in ('core_abs_pct', 'core_signed_pct', 'mae_L', 'drift_L', 'abs_drift_L',
                           'p99_L', 'p999_L', 'worst_core_pct', 'makespan_abs_pct',
                           'oracle_coverage', 'model_coverage'):
                row[metric] = float(rows[metric].mean())
            row['min_oracle_pairing'] = float(rows.oracle_coverage.min())
            row['paired_reads_total'] = int(rows.pairs.sum())
            result.append(row)
    return pd.DataFrame(result)


def available_foundation(frame, split):
    """Do not let a pending arm erase completed tests from other campaigns."""
    values = frame[(frame.cores == 1) & (frame.cohort == split)]
    return [arm for arm in FOUNDATION if (values.arm == arm).any()]


def foundation_numbers(frame):
    parts = [headline(frame[(frame.cores == 1) & (frame.cohort == split)],
                      available_foundation(frame, split)) for split in SPLITS]
    return pd.concat([p for p in parts if len(p)], ignore_index=True)


def foundation_coverage():
    rows = []
    for arm in FOUNDATION_SYNTH:
        campaign = arm.removesuffix('_single_core')
        availability = DATA.get('availability', {}).get(campaign, {})
        for split, expected in [('training',14), ('validation',14), ('test',24)]:
            values = DF[(DF.arm == arm) & (DF.cohort == split) & (DF.cores == 1)]
            count = len(values)
            rows.append(dict(arm=arm,cohort=split,completed_cases=count,expected_cases=expected,
                status='complete' if count == expected else
                       availability.get('single_core_test','unavailable') if split == 'test' else 'incomplete',
                schedule=availability.get('schedule','10+5')))
    table(pd.DataFrame(rows), 'foundation_coverage', caption='Available endpoint evidence (cases are not pooled across splits)')


def foundation():
    selected = DF[(DF.cores == 1) & DF.arm.isin(FOUNDATION)]
    foundation_coverage()
    numbers = foundation_numbers(selected)
    order = {v: i for i, v in enumerate(SPLITS)}
    numbers = numbers.assign(order=numbers.cohort.map(order)).sort_values(['order', 'arm']).drop(columns='order')
    table(numbers[['cohort', 'arm', 'cases', 'core_abs_pct', 'mae_L', 'p99_L',
                   'p999_L', 'oracle_coverage', 'model_coverage']], 'foundation_headlines',
          caption='Single-core results; equal weight per workload')
    for split in SPLITS:
        arms = available_foundation(selected, split)
        part = numbers[numbers.cohort == split].set_index('arm').loc[arms]
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7), sharey=True, layout='constrained')
        for ax, metric, title in zip(axes, ['core_abs_pct', 'mae_L'], ['Core timing', 'Request latency']):
            ax.barh(np.arange(len(part)), part[metric], color=[COLORS[a] for a in part.index], height=.65)
            for i, value in enumerate(part[metric]):
                ax.text(value + part[metric].max() * .02, i, f'{value:.2f}' if metric=='core_abs_pct' else f'{value:.3f}',
                        va='center', fontsize=8)
            ax.set_xlim(0, part[metric].max() * 1.22); ax.set_xlabel(PRETTY[metric]); ax.set_title(title)
            ax.grid(axis='x', alpha=.17); ax.set_axisbelow(True)
        axes[0].set_yticks(range(len(part)), [LABELS[a] for a in part.index]); axes[0].invert_yaxis()
        fig.suptitle(f'Single-core {split} · {int(part.cases.iloc[0])} workloads · equal workload weight', fontsize=10)
        finish(fig, 'foundation_headline_' + split)
    test = numbers[numbers.cohort == 'test'].set_index('arm')
    astra, deepseek, mess = [test.loc[a] for a in ['astra_single_core', 'deepseek_single_core', 'mess']]
    display(Markdown(
        f'On the 24 test workloads, Astra has **{astra.core_abs_pct:.2f}%** mean core error and '
        f'DeepSeek **{deepseek.core_abs_pct:.2f}%**, versus **{mess.core_abs_pct:.2f}%** for MeSS. '
        f'Their request MAE/L values are **{astra.mae_L:.3f}** and **{deepseek.mae_L:.3f}**, '
        f'versus **{mess.mae_L:.3f}**. These are outcomes of the selected campaigns, not '
        'replicated estimates of general LLM superiority. Lower core error does not guarantee '
        'lower request error on every workload.'))
    if 'opus_single_core' in test.index:
        opus = test.loc['opus_single_core']
        display(Markdown(f'Opus 5.5 has **{opus.core_abs_pct:.2f}%** mean core error and '
            f'**{opus.mae_L:.3f}** request MAE/L on the same 24 test workloads. '
            'This is a comparison of these frozen selections, not a general ranking of the providers.'))
    if 'gemini_single_core' not in test.index:
        display(Markdown('**Gemini: test evaluation pending.** Its completed training and validation '
            'results are included above. No test bar or zero-valued placeholder is plotted.'))
    cases=selected[(selected.cohort=='test')&(selected.arm=='mess')]
    google=int(cases['case'].str.contains(r'-c1-(?:sierra_a_|tahoe_)').sum())
    low=selected[(selected.arm=='mess')&selected.low_traffic]
    display(Markdown(
        f'The test headline weights {len(cases)} trace cases equally, including {google} Sierra/Tahoe '
        'Google trace regions; it is not a workload-family-balanced average. '
        + (f'{len(low)} single-core cases have fewer than 10,000 recorded eligible oracle reads; '
           'they remain flagged in the full tables.' if len(low) else
           'Every single-core case has at least 10,000 recorded eligible oracle reads.')))


def case_heatmaps(frame, arms, name, title):
    cases = sorted(set(frame['case']))
    width = max(9.3, 1.3 * len(arms) + 1.5) if any(a in arms for a in ('opus_single_core','gemini_single_core')) else 9.3
    fig, axes = plt.subplots(1, 2, figsize=(width, max(3.1, .265 * len(cases) + 1.4)),
                             sharey=True, layout='constrained')
    for ax, metric, cmap in zip(axes, ['core_signed_pct', 'mae_L'], ['RdBu_r', 'cividis']):
        pivot = frame.pivot(index='case', columns='arm', values=metric).reindex(index=cases, columns=arms)
        values = pivot.to_numpy(dtype=float)
        if metric == 'core_signed_pct':
            limit = max(1, float(np.nanmax(np.abs(values))))
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        else:
            norm = mpl.colors.Normalize(vmin=0, vmax=max(.01, float(np.nanmax(values))))
        image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect='auto')
        for (i, j), value in np.ndenumerate(values):
            if np.isfinite(value):
                rgba = image.cmap(image.norm(value)); light = .2126*rgba[0] + .7152*rgba[1] + .0722*rgba[2]
                ax.text(j, i, f'{value:+.1f}' if metric=='core_signed_pct' else f'{value:.2f}',
                        ha='center', va='center', fontsize=7, color='black' if light>.52 else 'white')
        ax.set_xticks(range(len(arms)), [LABELS[a] for a in arms], rotation=40, ha='right')
        ax.set_title(PRETTY[metric]); ax.tick_params(length=0)
        fig.colorbar(image, ax=ax, fraction=.035, pad=.025, shrink=.8)
    axes[0].set_yticks(range(len(cases)), [case_label(c) for c in cases])
    fig.suptitle(title, fontsize=11)
    finish(fig, name)


def convergence():
    frame = pd.DataFrame(DATA['convergence'])
    frame = frame[(frame.stage == 'single_core') & (frame.cores == 1)]
    fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True, layout='constrained')
    for j, split in enumerate(['training', 'validation']):
        for i, metric in enumerate(['core_abs_pct', 'mae_L']):
            ax = axes[i,j]
            for campaign in ('astra', 'deepseek', 'opus', 'gemini'):
                color = COLORS[campaign+'_single_core']
                for kind in ('submitted', 'retained'):
                    part = frame[(frame.campaign==campaign) & (frame.split==split) & (frame.kind==kind)].sort_values('round')
                    ax.plot(part['round'], part[metric], color=color, alpha=.45 if kind=='submitted' else 1,
                            linestyle=':' if kind=='submitted' else '-', marker='x' if kind=='submitted' else 'o',
                            markersize=3, linewidth=.85 if kind=='submitted' else 1.6)
            ax.set_ylabel(PRETTY[metric]); ax.grid(alpha=.17); ax.set_xticks(range(0,11,2))
            if i==0: ax.set_title(split.capitalize())
            else: ax.set_xlabel('Round (0 = common seed)')
    handles = [Line2D([],[],color=COLORS[a+'_single_core'],label='Opus 5.5' if a=='opus' else a.capitalize())
               for a in ('astra','deepseek','opus','gemini')]
    handles += [Line2D([],[],color='#555',label='Retained',marker='o',markersize=3),
                Line2D([],[],color='#999',label='Submitted',linestyle=':',marker='x',markersize=3)]
    fig.legend(handles=handles, loc='outside lower center', ncol=3, frameon=False)
    finish(fig, 'single_core_convergence')
    table(frame, 'convergence_single_core', caption='Convergence values', collapsed=True, presentation=False)


def outlier_inventory():
    test = DF[(DF.cohort=='test') & (DF.cores==1) & DF.arm.isin(FOUNDATION_SYNTH)]
    fig, ax = plt.subplots(figsize=(6.7,3.6), layout='constrained')
    for arm, marker in zip(FOUNDATION_SYNTH, ('o','s','^','D')):
        part = test[test.arm==arm]
        if part.empty: continue
        ax.scatter(part.core_abs_pct, part.mae_L, color=COLORS[arm], marker=marker,
                   s=27, alpha=.8, label=LABELS[arm])
    ax.set_xlabel('Absolute core-cycle error (%)'); ax.set_ylabel('Request MAE/L')
    ax.legend(frameon=False); ax.grid(alpha=.18)
    finish(fig, 'core_vs_request_error')
    selected = []
    for item in DATA['extreme_selection']:
        row = item['score']
        selected.append({**{k:row[k] for k in ('case','arm','core_abs_pct','mae_L','p99_L','p999_L','min_cycles','max_cycles','pairs','oracle_coverage')},
                         'Selection reason': '; '.join(item['reasons'])})
    table(pd.DataFrame(selected), 'selected_extremes', caption='Predeclared request-level case studies')
    ranked=[]
    for arm,group in test.groupby('arm',sort=False):
        group=group.copy()
        for metric,label in [('core_abs_pct','core_rank'),('mae_L','request_rank'),('p999_L','p999_rank')]:
            group[label]=group[metric].rank(method='min',ascending=False).astype(int)
        ranked.append(group[['case','arm','core_rank','request_rank','p999_rank','core_abs_pct','mae_L','p999_L',
                             'tail_1L_fraction','tail_1L_share','tail_5L_fraction','tail_5L_share']])
    table(pd.concat(ranked),'test_outlier_rankings',caption='All test-case rankings and tail contributions',collapsed=True)


def extreme_panel(item):
    score=item['score']; color=COLORS[item['arm']]; dist=item['histogram']
    display(Markdown('### '+LABELS[item['arm']]+' — '+case_label(item['case'])))
    fig, axes=plt.subplots(1,3,figsize=(10.3,2.65),layout='constrained')
    for owner,c in [('oracle','#333333'),('model',color)]:
        counts=np.asarray(dist[owner]);edges=np.asarray(dist['edges'])
        axes[0].stairs(counts/counts.sum(),edges,color=c,label='Oracle' if owner=='oracle' else 'Model')
    occupied=np.flatnonzero(np.asarray(dist['oracle'])+np.asarray(dist['model']))
    if len(occupied):axes[0].set_xlim(max(.5,edges[occupied[0]]*.9),edges[occupied[-1]+1]*1.1)
    axes[0].set_xscale('log');axes[0].set_xlabel('Recorded read latency (DRAM cycles)')
    axes[0].set_ylabel('Fraction per log-spaced bin');axes[0].legend(frameon=False)
    ccdf=item['ccdf']; x=np.asarray(ccdf['thresholds']);y=np.asarray(ccdf['probability']);keep=(x>0)&(y>0)
    axes[1].plot(x[keep],y[keep],color=color)
    axes[1].set(xscale='log',yscale='log',xlabel='Absolute request error / L',ylabel='Fraction exceeding threshold')
    for threshold in (1,5):axes[1].axvline(threshold,color='#aaa',linestyle=':',linewidth=.8)
    bins=pd.DataFrame(item['bins'])
    grid=np.round(np.arange(bins.instruction_million.min(),bins.instruction_million.max()+.05,.1),2)
    bins=bins.set_index('instruction_million').reindex(grid).rename_axis('instruction_million').reset_index()
    axes[2].fill_between(bins.instruction_million,bins.q01,bins.q99,color=color,alpha=.18,label='1st–99th percentile')
    axes[2].plot(bins.instruction_million,bins['mean'],color=color,label='Mean')
    axes[2].scatter(bins.instruction_million,bins.minimum,color=color,s=3,alpha=.5)
    axes[2].scatter(bins.instruction_million,bins.maximum,color=color,s=3,alpha=.5)
    axes[2].axhline(0,color='#555',lw=.6);axes[2].set_xlabel('Trace instruction ordinal (millions)')
    axes[2].set_ylabel('Signed request error / L');axes[2].legend(frameon=False,fontsize=6.5)
    for ax in axes:ax.grid(alpha=.14)
    fig.suptitle(f'{LABELS[item["arm"]]} · {case_label(item["case"])}',fontsize=10)
    finish(fig,'extreme_'+item['arm']+'_'+item['case'])
    signed=score['core_signed_pct']; tail=score['tail_5L_fraction']; contribution=score['tail_5L_share']
    display(Markdown(
        f"Core-cycle error is **{signed:+.2f}%**, while request MAE/L is **{score['mae_L']:.3f}**. "
        f"The paired extremes are **{score['min_cycles']:+,} / {score['max_cycles']:+,} DRAM cycles** "
        f"({score['min_L']:+.2f} / {score['max_L']:+.2f} L). "
        f"The median paired absolute error is **{item['absolute_error_quantiles_cycles']['0.5']:.1f} cycles**. "
        f"Errors above L affect {100*score['tail_1L_fraction']:.2f}% of pairs and contribute "
        f"{100*score['tail_1L_share']:.1f}% of total absolute error. "
        f"Errors above 5L affect **{100*tail:.3f}%** of paired reads and contribute "
        f"**{100*contribution:.2f}%** of total absolute paired error. "
        f"There are {score['pairs']:,} paired reads; oracle/model pairing coverage is "
        f"{100*score['oracle_coverage']:.2f}% / {100*score['model_coverage']:.2f}%. "
        'The first panel includes all recorded eligible reads; the other panels use trusted pairs. '
        'Dots in the last panel are per-bin extrema, not outlier-filtered values. '
        'These observations establish the error pattern, not which heuristic caused it.'))
    table(pd.DataFrame(item['extreme_requests']), 'requests_'+item['arm']+'_'+item['case'],
          caption='Exact extreme request records', collapsed=True, presentation=False)


def multicore():
    test=DF[DF.cohort=='test']
    numbers=headline(test,ALL_ARMS)
    all_splits=headline(DF,ALL_ARMS)
    table(all_splits,'endpoint_headlines_all_splits',caption='All cohort/core-count endpoint comparisons',collapsed=True)
    table(numbers[['cores','arm','cases','core_abs_pct','mae_L','worst_core_pct','makespan_abs_pct','p999_L','oracle_coverage']],
          'multicore_headlines',caption='Endpoint comparison and single-core retention')
    fig,axes=plt.subplots(2,2,figsize=(8.2,5.1),sharex=True,layout='constrained')
    for j,cores in enumerate([4,8]):
        part=numbers[numbers.cores==cores].set_index('arm').loc[ALL_ARMS]
        for i,metric in enumerate(['core_abs_pct','mae_L']):
            ax=axes[i,j]
            for k,arm in enumerate(ALL_ARMS):
                ax.bar(k,part.loc[arm,metric],color=COLORS[arm],alpha=.55 if arm.endswith('single_core') else 1,
                       hatch='//' if arm.endswith('multicore') else None,width=.7)
            ax.set_ylabel(PRETTY[metric]);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
            if i==0:ax.set_title(f'{cores} cores · 8 test mixes')
            else:ax.set_xticks(range(len(ALL_ARMS)),[LABELS[a] for a in ALL_ARMS],rotation=55,ha='right')
    finish(fig,'multicore_endpoint_comparison')
    for cores in (4,8):
        case_heatmaps(test[test.cores==cores],ALL_ARMS,f'multicore_{cores}_per_mix',f'{cores}-core test mixes')
    percore=[]
    for row in test[test.cores>1].to_dict('records'):
        for core,error in enumerate(row['per_core_signed_pct']):
            percore.append(dict(case=row['case'],arm=row['arm'],cores=row['cores'],core=core,
                                program=DATA['membership'][row['case']][core],signed_error_pct=error))
    percore=pd.DataFrame(percore)
    table(percore,'multicore_per_core',caption='All per-core signed cycle errors',collapsed=True)
    # Show every synthesized-model core, not only a favorable mix average.
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for ax,cores in zip(axes,(4,8)):
        cases=sorted(test.loc[test.cores==cores,'case'].unique())
        for k,arm in enumerate(SYNTH):
            part=percore[(percore.cores==cores)&(percore.arm==arm)]
            for j,case in enumerate(cases):
                values=part.loc[part['case']==case,'signed_error_pct']
                y=j+(k-1.5)*.17
                ax.scatter(values,np.repeat(y,len(values)),color=COLORS[arm],s=14,
                           marker='o' if arm.endswith('single_core') else 'x',
                           alpha=.55 if arm.endswith('single_core') else 1)
        ax.axvline(0,color='#777',lw=.8);ax.set_title(f'{cores} cores · all cores of all 8 mixes')
        ax.set_yticks(range(len(cases)),[case_label(c) for c in cases]);ax.invert_yaxis()
        ax.set_xlabel('Signed per-core cycle error (%)');ax.grid(axis='x',alpha=.15)
    handles=[Line2D([],[],color=COLORS[a],marker='o' if a.endswith('single_core') else 'x',
                    linestyle='none',label=LABELS[a]) for a in SYNTH]
    fig.legend(handles=handles,loc='outside lower center',ncol=4,frameon=False)
    finish(fig,'multicore_individual_cores')
    deltas=[]
    for campaign in ('astra','deepseek'):
        for cores in (1,4,8):
            a=numbers[(numbers.arm==campaign+'_single_core')&(numbers.cores==cores)].iloc[0]
            b=numbers[(numbers.arm==campaign+'_multicore')&(numbers.cores==cores)].iloc[0]
            deltas.append(dict(campaign=campaign,cores=cores,core_change_pp=b.core_abs_pct-a.core_abs_pct,
                               request_MAE_L_change=b.mae_L-a.mae_L))
    table(pd.DataFrame(deltas),'multicore_changes',caption='MC minus SC selection; negative means improvement',presentation=False)
    lines=[]
    for campaign in ('astra','deepseek'):
        eight=numbers[numbers.cores==8].set_index('arm')
        a,b=[eight.loc[campaign+'_'+stage] for stage in ('single_core','multicore')]
        lines.append(f'{campaign.capitalize()}: eight-core error changes from {a.core_abs_pct:.2f}% to '
                     f'{b.core_abs_pct:.2f}%; request MAE/L changes from {a.mae_L:.3f} to {b.mae_L:.3f}.')
    mess=numbers[(numbers.cores==8)&(numbers.arm=='mess')].iloc[0]
    display(Markdown(' '.join(lines)+f' MeSS remains competitive on request accuracy ({mess.mae_L:.3f} at eight cores). '
        'The extra rounds improve average core timing, not every latency metric. The per-core figure '
        'also prevents a mix mean from hiding a poorly modeled program.'))


def hardware():
    frame=pd.DataFrame([r for r in DATA['transfer']['rows'] if r['study']=='hardware'])
    frame['organization']=frame.point.str.rsplit('_q',n=1).str[0]
    frame['queue']=frame.point.str.rsplit('_q',n=1).str[1].astype(int)
    orgs=['DDR5_16Gb_x8','DDR5_8Gb_x8','DDR5_16Gb_x16']
    names=['8 BG × 4 banks · 8 GiB · ×8','8 BG × 2 banks · 4 GiB · ×8','4 BG × 4 banks · 4 GiB · ×16']
    aggregated=frame.groupby(['cores','organization','queue','arm'],sort=False).agg(
        cases=('case','size'),core_abs_pct=('core_abs_pct','mean'),mae_L=('request_mae_over_L','mean'),
        p999_L=('p999_over_L','mean'),oracle_coverage=('oracle_coverage','mean')).reset_index()
    for cores in (1,4,8):
        fig,axes=plt.subplots(2,3,figsize=(9,4.7),sharex=True,sharey='row',layout='constrained')
        for j,(org,title) in enumerate(zip(orgs,names)):
            for i,metric in enumerate(['core_abs_pct','mae_L']):
                ax=axes[i,j]
                for arm in SYNTH:
                    g=aggregated[(aggregated.cores==cores)&(aggregated.organization==org)&(aggregated.arm==arm)].sort_values('queue')
                    ax.plot(g['queue'].astype(str),g[metric],color=COLORS[arm],label=LABELS[arm],
                            linestyle='--' if arm.endswith('multicore') else '-',
                            marker='s' if arm.startswith('deepseek') else 'o',markersize=4)
                if i==0:ax.set_title(title,fontsize=9)
                if i==1:ax.set_xlabel('Read/write queue depth (entries)')
                if j==0:ax.set_ylabel(PRETTY[metric])
                ax.grid(alpha=.17)
        fig.suptitle(f'Configuration transfer · {cores} '+('core' if cores==1 else 'cores'),fontsize=11)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
        finish(fig,f'hardware_{cores}core')
    table(aggregated,'hardware_headlines',caption='All organization/queue/core-count headlines',collapsed=True)
    # Workload heatmaps for every organization/queue, one per core count/endpoint.
    for cores in (1,4,8):
        for arm in SYNTH:
            group=frame[(frame.cores==cores)&(frame.arm==arm)]
            columns=[org+f'_q{q}' for org in orgs for q in (32,64,128)]
            values=group.pivot(index='case',columns='point',values='core_abs_pct').reindex(columns=columns).sort_index()
            fig,ax=plt.subplots(figsize=(7.8,max(3,.23*len(values)+1.4)),layout='constrained')
            image=ax.imshow(values.to_numpy(),cmap='cividis',aspect='auto',vmin=0,
                            vmax=float(frame.loc[frame.cores==cores,'core_abs_pct'].max()))
            ax.set_xticks(range(9),[str(q) for _ in orgs for q in (32,64,128)])
            ax.set_xlabel('Read/write queue depth (entries)')
            group_axis=ax.secondary_xaxis('top')
            group_axis.set_xticks([1,4,7],['32 banks · ×8 · 8 GiB','16 banks · ×8 · 4 GiB','16 banks · ×16 · 4 GiB'])
            group_axis.tick_params(length=0,labelsize=8,pad=5)
            group_axis.spines['top'].set_visible(False)
            ax.vlines([2.5,5.5],-.5,len(values)-.5,color='white',linewidth=1)
            ax.set_yticks(range(len(values)),[case_label(x) for x in values.index]);ax.tick_params(length=0)
            fig.colorbar(image,ax=ax,label='Mean absolute core error (%)',fraction=.03,pad=.02)
            ax.set_title(f'{LABELS[arm]} · {cores} '+('core' if cores==1 else 'cores')+' · configuration transfer',pad=30)
            # Keep detailed figures in the notebook too, without overwhelming the main narrative.
            fig.canvas.draw();fig.set_layout_engine('none')
            for suffix in ('pdf','svg','png'):fig.savefig(FIGURES/f'hardware_detail_{cores}_{arm}.{suffix}',bbox_inches='tight')
            preview=io.BytesIO();fig.savefig(preview,format='png',dpi=130,bbox_inches='tight')
            encoded=base64.b64encode(preview.getvalue()).decode()
            display(HTML(f'<details><summary>{html.escape(LABELS[arm])}: {cores}-core per-case configuration heatmap</summary>'
                         f'<img alt="Per-case configuration-transfer core errors" style="max-width:100%" '
                         f'src="data:image/png;base64,{encoded}"></details>'))
            plt.close(fig)
    table(frame,'hardware_per_case',caption='Complete configuration-transfer results',collapsed=True,presentation=False)
    eight=aggregated[aggregated.cores==8]
    changed=eight[(eight.organization=='DDR5_16Gb_x16')&(eight.queue==32)].set_index('arm')
    default=eight[(eight.organization=='DDR5_16Gb_x8')&(eight.queue==64)].set_index('arm')
    a,d=changed.loc['astra_multicore'],changed.loc['deepseek_multicore']
    dsc=changed.loc['deepseek_single_core']
    display(Markdown(
        f'The most constrained ×16/Q32 eight-core setting exposes a limitation hidden by the default results: '
        f'Astra MC reaches **{a.core_abs_pct:.2f}%** core error and DeepSeek MC **{d.core_abs_pct:.2f}%**, '
        f'versus {default.loc["astra_multicore","core_abs_pct"]:.2f}% and '
        f'{default.loc["deepseek_multicore","core_abs_pct"]:.2f}% at the default organization/Q64. '
        f'DeepSeek SC is better than its MC successor here ({dsc.core_abs_pct:.2f}%). '
        'Adaptation to default-geometry contention therefore does not guarantee configuration robustness. '
        'DeepSeek’s fixed decoder/timings are a concrete source limitation, but these experiments do not '
        'isolate their contribution from queueing, timing and placement changes. Detailed heatmaps share '
        'a color scale within each core count.'))


def gem5():
    frame=pd.DataFrame([r for r in DATA['transfer']['rows'] if r['study']=='gem5'])
    frame['mode']=np.where(frame['case'].str.startswith('polybench'),'SE','FS')
    frame['makespan_abs_pct']=frame.core_abs_pct
    table(frame.groupby(['mode','ending','arm']).agg(cases=('case','size'),time_error_pct=('core_abs_pct','mean'),
          signed_error_pct=('signed_pct','mean')).reset_index(),'gem5_headlines',
          caption='gem5: separate execution modes and interval endings')
    cases=sorted(frame['case'].unique(),key=lambda n:(not n.startswith('polybench'),n))
    fig,ax=plt.subplots(figsize=(9,max(4,.27*len(cases)+1.2)),layout='constrained')
    pivot=frame.pivot(index='case',columns='arm',values='signed_pct').reindex(index=cases,columns=ALL_ARMS)
    maximum=max(1,float(np.abs(pivot.to_numpy()).max()))
    image=ax.imshow(pivot.to_numpy(),cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-maximum,vcenter=0,vmax=maximum),aspect='auto')
    for (i,j),value in np.ndenumerate(pivot.to_numpy()):
        rgb=image.cmap(image.norm(value));light=.2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2]
        ax.text(j,i,f'{value:+.1f}',ha='center',va='center',fontsize=7,color='black' if light>.52 else 'white')
    ending=frame.groupby('case')['ending'].first()
    def name(c):
        if c.startswith('polybench421_'):return 'SE · '+c.removeprefix('polybench421_').removesuffix('_large').upper()
        suite,work=c.split('-c1-')
        titles={'gapbs':'GAP','npb':'NPB','parsec':'PARSEC'}
        workload={'bfs':'BFS','pr':'PageRank','sssp':'SSSP','blackscholes':'Black-Scholes',
                  'canneal':'Canneal','streamcluster':'Streamcluster'}.get(work,work.removesuffix('.x').upper())
        return 'FS · '+titles[suite]+' '+workload
    labels=[name(c) +
            (' [2B cap]' if ending[c]=='instruction_cap' else ' [ROI end]') for c in cases]
    ax.set_yticks(range(len(cases)),labels)
    ax.set_xticks(range(len(ALL_ARMS)),[LABELS[a] for a in ALL_ARMS],rotation=45,ha='right')
    ax.tick_params(length=0);fig.colorbar(image,ax=ax,label='Signed measured-time error (%)',fraction=.03,pad=.02)
    ax.set_title('Frontend transfer · gem5 O3 · single core',pad=12)
    finish(fig,'gem5_per_workload')
    intervals=frame.groupby('case').agg(mode=('mode','first'),ending=('ending','first'),
        oracle_instructions=('oracle_instructions','first'),min_model_instructions=('instructions','min'),
        max_model_instructions=('instructions','max'),minimum_recorded_reads=('dram_reads','min'),
        low_traffic=('low_traffic','any')).reindex(cases).reset_index()
    table(intervals,'gem5_intervals',caption='Actual interval sizes; instruction counts are not inferred from targets',
          collapsed=True,presentation=False)
    columns=['case','arm','mode','ending','core_abs_pct','signed_pct','instructions','oracle_instructions',
             'cycles','ipc','dram_reads','verification','low_traffic','roi_seconds','total_hours','peak_rss_gib']
    table(frame[columns],'gem5_per_case',caption='All gem5 intervals, traffic and verification',collapsed=True,presentation=False)
    summary=frame.groupby(['mode','ending','arm']).core_abs_pct.mean()
    fs_cap={a:summary.loc['FS','instruction_cap',a] for a in SYNTH+['mess']}
    worst=frame[frame.arm.isin(SYNTH)].sort_values('core_abs_pct',ascending=False).iloc[0]
    display(Markdown(
        f'On the five instruction-capped FS workloads, mean timing error is '
        f'**{fs_cap["astra_single_core"]:.2f}% / {fs_cap["astra_multicore"]:.2f}%** for Astra SC/MC '
        f'and **{fs_cap["deepseek_single_core"]:.2f}% / {fs_cap["deepseek_multicore"]:.2f}%** for DeepSeek, '
        f'versus **{fs_cap["mess"]:.2f}%** for MeSS. The largest synthesized-model error anywhere in '
        f'this gem5 cohort is **{worst.core_abs_pct:.2f}%**, on `{worst["case"]}` with {LABELS[worst.arm]}. '
        'These are transfer results for the recorded intervals—not full-program results for the capped cases. '
        'Completed, short SE kernels and capped FS intervals should not be collapsed into one headline.'))


def lat_tp_status(study):
    """Validate display populations; unfinished points must not acquire metrics."""
    frame = pd.DataFrame(study['point_statuses'])
    keys = ['arm', 'read_ratio', 'nop_counter', 'streaming_only']
    if frame.duplicated(keys).any():
        raise ValueError('duplicate Lat–Tp point')
    complete = frame.status == 'complete'
    metrics = ['throughput_GBps', 'probe_latency_ns']
    if frame.loc[~complete, metrics].notna().any().any():
        raise ValueError('unfinished Lat–Tp point contains scored metrics')
    for column, population in [('throughput_GBps', complete),
                               ('probe_latency_ns', complete & ~frame.streaming_only)]:
        values = frame.loc[population, column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError('invalid completed Lat–Tp metric')
    if int(complete.sum()) != study['successful'] or len(frame) != study['expected']:
        raise ValueError('Lat–Tp coverage does not reconcile')
    return frame


def lat_tp_curve(frame, arm, ratio, nops):
    """Leave a gap at every failed or unmeasured offered load, not a fitted line."""
    subset = frame[(frame.arm == arm) & (frame.read_ratio == ratio) & ~frame.streaming_only]
    return subset.set_index('nop_counter').reindex(sorted(nops, reverse=True))[
        ['throughput_GBps', 'probe_latency_ns']].astype(float)


def lat_tp(extension=False):
    study=DATA.get('lat_tp')
    if not study:
        display(Markdown('The refreshed Lat–Tp study is not yet included. '
                        'No historical curve or incomplete point is substituted.'))
        return
    arms=['oracle']+ (SYNTH if extension else LAT_TP_FOUNDATION)
    status = lat_tp_status(study)
    name = 'lat_tp_endpoint' if extension else 'lat_tp_foundation'
    coverage = pd.crosstab(status.arm, status.status).reindex(
        index=arms, columns=['complete', 'failed', 'incomplete', 'pending'], fill_value=0).fillna(0).astype(int)
    coverage.index.name = 'arm'
    table(coverage.reset_index(), name+'_coverage', caption='Lat–Tp coverage (157 points per model)')
    display(Markdown(
        f'Across the nine-model study, **{study["successful"]:,}/{study["expected"]:,} points completed**, '
        f'{study["failed"]:,} failed, {study.get("incomplete", 0):,} are incomplete and '
        f'{study["pending"]:,} are pending. Failed and missing points have no metric. '
        f'{study.get("failed_attempt_count", 0):,} failed attempts are retained, including earlier '
        'attempts whose points may since have completed.'))
    if study.get('pending', 0) or study.get('running', False):
        display(Markdown('**In-progress snapshot.** These are the completed measurements available '
                         'when this notebook was generated, not a claim that the full sweep has finished. '
                         'Here, “failed” means a failure has been recorded; an authorized higher-memory '
                         'retry can still be running. All non-complete points remain unscored.'))
    policy = study.get('execution_policy')
    if policy:
        display(Markdown(
            'The four original high-load FixedLat/MeSS attempts exhausted a 4-GiB process limit. '
            f'The resource-only retry allows {policy["high_memory_bytes"]/2**30:g} GiB for these '
            f'two baselines, with a {policy["timeout_seconds"]/60:g}-minute limit per simulation. '
            'Successful earlier points are retained. New runs omit request CSV logs after exact '
            'counter-parity checks; the frozen models and traffic are unchanged.'))
    phase = study.get('execution', {}).get('phase', 'prepared')
    if phase not in {'sweep', 'complete', 'complete_with_failures'} and study.get('pending', 0):
        display(Markdown('**Qualification points only; a full sweep is not yet recorded here.** '
                         'The following table does not imply curves between the sampled corners.'))
        corners=status[status.arm.isin(arms)&(status.status!='pending')].copy()
        table(corners[['arm','read_ratio','nop_counter','streaming_only','status','throughput_GBps',
                       'probe_latency_ns','reason']],
              'lat_tp_extension_qualification' if extension else 'lat_tp_foundation_qualification',
              caption='Completed and failed qualification corners; not a sweep',collapsed=True)
        return
    display(Markdown(
        'Curves retain each adapter’s native admission and posted-write semantics. These are not '
        'equal-buffer comparisons: the oracle and synthesized-model wrapper have finite queues, '
        'whereas the baseline paths can admit unbounded traffic. Reported throughput above the '
        'channel’s physical rate is an adapter/model limitation, not achievable DRAM bandwidth. '
        'Both axes use log scales so these excursions remain visible. Lines follow offered-load '
        'order, break at missing points, and never extrapolate.'))
    nops = study['protocol']['case']['suite']['nop_counters']
    fig,axes=plt.subplots(2,3,figsize=(9.6,5.8),sharex=True,sharey=True)
    partial = int(coverage.pending.sum() + coverage.incomplete.sum()) > 0
    fig.subplots_adjust(left=.085,right=.99,top=.91 if partial else .94,bottom=.21,hspace=.27,wspace=.10)
    if partial:
        fig.suptitle(f'Partial sweep · {int(coverage.complete.sum())}/{int(coverage.to_numpy().sum())} '
                     'points complete',y=.995,fontsize=10)
    for ax,ratio in zip(axes.flat,[100,90,80,70,60,50]):
        for arm in arms:
            part=lat_tp_curve(status,arm,ratio,nops)
            ax.plot(part.throughput_GBps,part.probe_latency_ns,color=COLORS[arm],label=LABELS[arm],
                    linestyle='--' if arm.endswith('multicore') else '-',marker='.',markersize=3,linewidth=1.2)
        ax.set_title(f'{ratio}% streaming reads')
        ax.set_xscale('log');ax.set_yscale('log');ax.grid(alpha=.16)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.54,.005),ncol=4,frameon=False)
    fig.supxlabel('Reported total DRAM throughput (GB/s)',y=.115)
    fig.supylabel('Mean dependent-read probe latency (ns)',x=.012)
    finish(fig,'lat_tp_endpoint_changes' if extension else 'lat_tp_foundation')
    unloaded=status[(~status.streaming_only)&(status.read_ratio==100)&(status.nop_counter==max(nops))
                    ].set_index('arm').reindex(arms)
    streaming=status[status.streaming_only].set_index('arm').reindex(arms)
    summary=pd.DataFrame({'arm':arms,'Low-load probe latency (ns)':unloaded.probe_latency_ns.to_numpy(),
        'Low-load status':unloaded.status.to_numpy(),'Pure-read streaming (GB/s)':streaming.throughput_GBps.to_numpy(),
        'Streaming status':streaming.status.to_numpy()})
    table(summary,name+'_summary',caption='Lowest offered streaming load (100% reads) and pure streaming; missing is not zero')
    columns=['arm','read_ratio','nop_counter','streaming_only','status','throughput_GBps','probe_latency_ns','reason']
    table(status[columns],'lat_tp_all_points',caption='Every Lat–Tp load point, including failures',collapsed=True,presentation=False)


def appendices():
    for split in SPLITS:
        frame=DF[(DF.cohort==split)&(DF.cores==1)&DF.arm.isin(FOUNDATION)].sort_values(['case','arm'])
        table(frame[['case','arm','core_abs_pct','core_signed_pct','mae_L','drift_L','p99_L','p999_L',
                     'min_cycles','max_cycles','L','pairs','oracle_coverage','model_coverage']],
              f'single_core_{split}_per_case',caption=f'Complete {split} single-core breakdown',collapsed=True)
    table(DF,'all_default_configuration_scores',caption='Complete default-configuration evidence',collapsed=True,presentation=False)
    membership=pd.DataFrame([{'case':case,'core_order':', '.join(programs)} for case,programs in DATA['membership'].items()])
    table(membership,'membership',caption='Workloads and ordered mix membership',collapsed=True)
    coverage=pd.DataFrame(DATA['transfer']['coverage'])
    table(coverage.groupby(['study','cores','state']).size().rename('Measurements').reset_index(),
          'transfer_coverage',caption='Transfer-study coverage',presentation=False)
    host_times=coverage.groupby(['host','study','cores']).agg(measurements=('id','size'),
                        median_wall_s=('process_wall_seconds','median'),maximum_wall_s=('process_wall_seconds','max')).reset_index()
    table(host_times,'host_runtime',caption='Host-separated concurrent-run timings (not speed benchmarks)',collapsed=True,presentation=False)
    table(pd.DataFrame(DATA['oracle_equivalence']),'oracle_equivalence',caption='Exact oracle-equivalence checks',collapsed=True,presentation=False)
    decisions=pd.DataFrame(DATA['decisions'])
    decisions['decision']=decisions.decision.map(lambda x:json.dumps(x,ensure_ascii=False))
    table(decisions,'promotion_decisions',caption='Recorded promotion decisions, unchanged',collapsed=True,presentation=False)
    for arm,record in DATA['sources'].items():
        numbered='\n'.join(f'{i:3}  {line}' for i,line in enumerate(record['source'].splitlines(),1))
        display(HTML(f'<details id="source-{arm}"><summary>{html.escape(LABELS[arm])}: frozen source, '
                     f'SHA256 {record["candidate"]["files"]["model.cpp"]["sha256"]}</summary>'
                     f'<pre>{html.escape(numbered)}</pre></details>'))
        display(HTML(f'<details><summary>{html.escape(LABELS[arm])}: exact parameter overrides</summary>'
                     f'<pre>{html.escape(json.dumps(record["candidate"]["parameters"],indent=2))}</pre></details>'))
