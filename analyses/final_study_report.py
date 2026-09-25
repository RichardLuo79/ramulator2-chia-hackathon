"""Compact, reproducible analysis of the frozen final-study evidence.

Reads verified receipts and comparisons. It never runs a simulator, changes a
model, or treats an incomplete measurement as zero error.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path


COLORS={'astra_single_core':'#0072B2','astra_multicore':'#56B4E9',
        'deepseek_single_core':'#D55E00','deepseek_multicore':'#E69F00',
        'gemini_single_core':'#009E73','gemini_multicore':'#CC79A7',
        'fixedlat':'#777777','md1':'#333333','wmg1':'#882255','mess':'#AA4499'}
LABELS={'astra_single_core':'Astra\nSC endpoint','astra_multicore':'Astra\nMC endpoint',
        'deepseek_single_core':'DeepSeek\nSC endpoint','deepseek_multicore':'DeepSeek\nMC endpoint',
        'gemini_single_core':'Gemini\nSC endpoint','gemini_multicore':'Gemini\nMC endpoint',
        'fixedlat':'Queued\nFixedLat','md1':'M/D/1','wmg1':'WMG1','mess':'MeSS'}


def read(path):return json.loads(path.read_text())


def flattened(job,comparison):
    row={key:job.get(key) for key in ('id','study','case','arm','cores','point')}
    row['accepted']=comparison.get('accepted',False)
    row['reason']=comparison.get('reason')
    row['host']=(comparison.get('qualified_hosts') or [None])[0]
    if not row['accepted']:return row
    if job['study']=='gem5':
        model=comparison['model'];oracle=comparison['oracle']
        row.update(core_abs_pct=comparison['absolute_time_error_pct'],
            signed_pct=comparison['signed_time_error_pct'],ending=model['stop_reason'],
            instructions=model['instructions'],cycles=model['cycles'],ipc=model['ipc'],
            dram_reads=model['dram_reads_admitted'],low_traffic=model['low_traffic'],
            verification=model['verification'],roi_seconds=model['roi_wall_seconds'],
            total_hours=model['total_wall_hours'],oracle_instructions=oracle['instructions'],
            cpu_seconds=model['roi_cpu_seconds'],peak_rss_gib=model['max_rss_gib'],
            initialization_seconds=model['initialization_seconds'],setup_seconds=model['setup_seconds'],
            warmup_seconds=model['warmup_seconds'],verification_seconds=model['verification_seconds'])
    else:
        score=comparison['score'];cycles=score['cycles'];request=score['request']
        row.update(core_abs_pct=cycles['mean_abs_per_core_pct'],
            signed_pct=cycles['legacy_signed_mean_pct'],
            worst_core_pct=max(abs(v) for v in cycles['per_core_dev_pct']),
            per_core_signed_pct=cycles['per_core_dev_pct'],makespan_signed_pct=cycles['makespan_dev_pct'],
            request_mae_over_L=request['mae'],request_signed_over_L=request['sgn'],
            request_abs_drift_over_L=abs(request['sgn']),p99_over_L=request['tail'],
            p999_over_L=request.get('p999_over_L'),negative_cycles=request['extreme_min_cycles'],
            positive_cycles=request['extreme_max_cycles'],negative_over_L=request['extreme_min_over_L'],
            positive_over_L=request['extreme_max_over_L'],L=request['oracle_read_mean_latency'],
            oracle_coverage=request['cov_o'],model_coverage=request['cov_m'],pairs=request['matched'],
            address_mismatches=score['request_pairing']['address_mismatch_pairs'],
            low_traffic=score['oracle_owner_reads']<10000,
            pairing=score['request_pairing'],tails=request.get('tail_thresholds'))
    return row


def report(root,output):
    import nbformat
    from nbclient import NotebookClient
    manifest_path=root/'waves/gemini/manifest.json'
    if not manifest_path.exists():manifest_path=root/'manifest.json'
    manifest=read(manifest_path);jobs={j['id']:j for j in manifest['jobs']}
    rows=[]
    for path in sorted((root/'collected-comparisons').glob('*.json')):
        value=read(path)
        if value['id'] in jobs:rows.append(flattened(jobs[value['id']],value))
    expected=defaultdict(list)
    for job in manifest['jobs']:
        expected[job['study']].append(job['arm'])
    expected={k:sorted(set(v)-{'oracle'}) for k,v in expected.items()}
    # A receipt with complete=True still waits for that host's cross-host check.
    coverage=[]
    for job in manifest['jobs']:
        path=root/'collected-status'/(job['id']+'.json')
        receipt=read(path) if path.exists() else None
        observation=(receipt or {}).get('result',{}).get('observation',{})
        coverage.append({**{k:job.get(k) for k in ('id','study','arm','case','cores','point')},
            'state':'pending' if receipt is None else ('completed' if receipt['complete'] else 'failed'),
            'error':None if receipt is None else receipt.get('error'),
            'host':None if receipt is None else receipt.get('host'),
            'process_wall_seconds':observation.get('process',{}).get('wall_seconds'),
            'per_core_instructions':observation.get('frontend_stats',{}).get('per_core_instructions'),
            'background':observation.get('frontend_stats',{}).get('background'),
            'admission_windows':observation.get('frontend_stats',{}).get('admission_windows')})
    sources={name:dict(candidate=value['candidate'],stage=value['stage']) for name,value in manifest['endpoints'].items()}
    data=dict(rows=rows,coverage=coverage,expected_models=expected,endpoints=sources,
        source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        hardware_points=manifest['hardware_points'],cutoff=manifest['stop_utc'],
        case_membership=read(root/'baseline-settings.json')['evaluation']['champsim']['cases'],
        initial_matrix_jobs=len(jobs),gemini_pending=not any('gemini' in name for name in sources))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    if rows:
        fields=sorted(set().union(*(r.keys() for r in rows)))
        with output.with_suffix('.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)
    intro='''# Final-model transfer and hardware generalization

This notebook embeds its compact evidence; opening it needs no cloud connection
or raw trace archive. Missing and failed results remain missing. Results are
provisional while the coverage table is incomplete. Gemini is added only after
its stage selections freeze.

## Protocol and weighting

**gem5:** single-core O3, polling Ramulator integration, 2M committed instructions
of warmup inside the ROI, followed by up to 2B more instructions or ROI end.
The error is `100 × (Tmodel − Toracle) / Toracle`; headline error is its absolute
value. Naturally ended and instruction-capped intervals are separate. No
per-request matching is attempted in gem5.

**ChampSim:** the existing test cohort (24 single-core workloads and eight mixes
at each of four/eight cores); 2M warmup + 20M measured instructions per core.
Completed cores continue as background traffic and replay their traces. Their
measured counters stay frozen. Only reads admitted within each source core's
measurement window are scored; there is no extra drain. Physical placement is
timing-independent and capacity-correct, shared across models.

For each mix, core error averages the absolute percentage errors of its cores.
Headlines weight cases equally, never by instruction or request count. Request
error is `d = latency_model − latency_oracle`, in DRAM cycles. `L` is the mean
latency of **all recorded eligible oracle reads**, including unpaired reads.
MAE/L and tail percentiles use paired reads; missing/unidentifiable reads remain
explicitly unknown. A small signed error can hide cancellation.

Queue depths 32/64/128 are crossed with 32-bank 8-GiB x8, 16-bank 4-GiB x8, and
16-bank 4-GiB x16 DDR5-4800 organizations. Geometry, capacity and derived timings
change together: these are configuration-transfer experiments, not isolated
causal bank-count tests. Warmup bypasses Ramulator, so its controller starts cold.

New comparisons use corrected baseline sources. Historical campaign feedback
is not rewritten or pooled into these measurements. The test cohort was
previously examined by the research team. No tuning or agent feedback occurs.
'''
    cells=[nbformat.v4.new_markdown_cell(intro),nbformat.v4.new_code_cell(
        'import json, math\nfrom pathlib import Path\nimport pandas as pd\nimport numpy as np\n'
        'import matplotlib.pyplot as plt\nfrom IPython.display import display\n'
        'data = json.loads('+repr(json.dumps(data,allow_nan=False))+')\n'
        'colors = '+repr(COLORS)+'\n'
        'labels = '+repr(LABELS)+'\n'
        "plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'svg.fonttype':'none'})\n"
        "figures=Path('figures/final_study'); figures.mkdir(parents=True,exist_ok=True)\n"
        "df=pd.DataFrame(data['rows']); coverage=pd.DataFrame(data['coverage'])\n"
        "display(coverage.groupby(['study','cores','state']).size().unstack(fill_value=0))\n"
        "print('Gemini selections pending:',data['gemini_pending'])\n"
        "display(coverage.loc[coverage.state=='failed',['study','case','arm','error']])"),
        nbformat.v4.new_markdown_cell('## Shared-cohort headline comparisons\n\nOnly cases with every required model accepted contribute to a headline. Coverage is shown alongside the means. Individual partial results remain available below.'),
        nbformat.v4.new_code_cell('''headlines=[]
if not df.empty:
    valid=df[df.accepted].copy()
    valid['point']=valid['point'].fillna('default')
    valid['ending']=valid.get('ending',pd.Series(index=valid.index,dtype=object)).fillna('fixed-window')
    for (study,cores,point,ending),group in valid.groupby(['study','cores','point','ending']):
        required=set(data['expected_models'][study])
        shared=[case for case,g in group.groupby('case') if required <= set(g.arm)]
        if not shared: continue
        common=group[group.case.isin(shared)]
        columns=[c for c in ['core_abs_pct','signed_pct','worst_core_pct','request_mae_over_L','p99_over_L','p999_over_L','oracle_coverage'] if c in common and common[c].notna().any()]
        table=common.groupby('arm')[columns].mean().reset_index()
        table=table.assign(study=study,cores=cores,point=point,ending=ending,cases=len(shared))
        headlines.append(table)
headline=pd.concat(headlines,ignore_index=True) if headlines else pd.DataFrame()
display(headline if not headline.empty else 'No complete shared cohort yet; no headline is reported.')
if not headline.empty:
    for (study,cores,point,ending),g in headline.groupby(['study','cores','point','ending']):
        fig,ax=plt.subplots(figsize=(9,3.4))
        ax.bar([labels.get(x,x) for x in g.arm],g.core_abs_pct,color=[colors.get(x,'#777777') for x in g.arm])
        ax.set_ylabel('Mean absolute core/time error (%)'); ax.tick_params(axis='x',labelsize=9)
        ax.set_title(f'{study} · {cores} core(s) · {point} · {ending} · {int(g.cases.iloc[0])} shared cases')
        fig.tight_layout(); fig.savefig(figures/f'{study}-{cores}-{point}-{ending}.svg'); plt.show()
'''),nbformat.v4.new_markdown_cell('## Per-case errors and extreme cases\n\nRows with unavailable measurements are not colored as zero. Core and request errors are different outcomes; the largest request error is not automatically the cause of the largest CPU slowdown.'),
        nbformat.v4.new_code_cell('''if not df.empty:
    valid=df[df.accepted].copy()
    display(valid.sort_values('core_abs_pct',ascending=False).head(20))
    for (study,cores),g in valid.groupby(['study','cores']):
        if study=='hardware':g=g[g.point=='DDR5_16Gb_x8_q64']
        if g.empty:continue
        table=g.pivot(index='case',columns='arm',values='core_abs_pct')
        fig,ax=plt.subplots(figsize=(max(7,.9*len(table.columns)),max(3,.3*len(table))))
        image=ax.imshow(np.ma.masked_invalid(table.to_numpy()),aspect='auto',cmap='cividis')
        ax.set_xticks(range(len(table.columns)),[labels.get(x,x) for x in table.columns],rotation=25,ha='right')
        ax.set_yticks(range(len(table)),table.index)
        ax.set_title(f'{study}: {cores}-core default-configuration per-case error')
        fig.colorbar(image,ax=ax,label='Absolute core/time error (%)')
        fig.tight_layout();fig.savefig(figures/f'heatmap-{study}-{cores}.svg');plt.show()
    hardware=valid[valid.study=='hardware']
    if not hardware.empty:
        display(hardware.sort_values('p999_over_L',ascending=False).head(20))
        for cores,g in hardware.groupby('cores'):
            fig,ax=plt.subplots(figsize=(7,4))
            for arm,h in g.groupby('arm'):
                ax.scatter(h.core_abs_pct,h.request_mae_over_L,label=arm,color=colors.get(arm),alpha=.65,s=22)
            ax.set_xlabel('Mean absolute core error (%)');ax.set_ylabel('Paired request MAE/L')
            ax.set_title(f'{cores}-core cases across all nine configurations');ax.legend(fontsize=8)
            fig.tight_layout();fig.savefig(figures/f'core-versus-request-{cores}.svg');plt.show()
        display(hardware[['case','arm','cores','point','negative_cycles','positive_cycles','negative_over_L','positive_over_L','oracle_coverage','pairs']].sort_values('positive_cycles',ascending=False))
    display(valid)
'''),nbformat.v4.new_markdown_cell('## Provenance and limits\n\nHost wall times are descriptive concurrent-run timings, not isolated simulator-speed benchmarks. Failures and unfinished intervals cannot establish accuracy. Source inspection identified DeepSeek\'s fixed bank decoder; the sweep tests the resulting behavior without changing it. See `final_model_review_0923.md` for mechanisms and unresolved hypotheses.'),
        nbformat.v4.new_code_cell("display(pd.DataFrame([{'endpoint':k,**v['candidate']} for k,v in data['endpoints'].items()]))\nprint('Frozen job manifest SHA256:',data['source_manifest_sha256'])\ndisplay(pd.DataFrame([{'case':k,'programs_in_core_order':v['programs']} for k,v in data['case_membership'].items()]))\ndisplay(coverage.groupby(['host','study','state'],dropna=False).size().rename('jobs').reset_index())\ndisplay(coverage.groupby(['host','study','cores'],dropna=False)['process_wall_seconds'].agg(['count','median','max']))\ndisplay(coverage[['case','arm','point','per_core_instructions','background','admission_windows']])")]
    notebook=nbformat.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')))
    NotebookClient(notebook,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(output.parent)}}).execute()
    nbformat.write(notebook,output)
    return dict(comparisons=len(rows),required=len(jobs),notebook=str(output))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();print(json.dumps(report(args.root,args.output)))
