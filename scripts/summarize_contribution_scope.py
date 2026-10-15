#!/usr/bin/env python3
"""Descriptive all-fault and leave-one-fault-out summaries of frozen losses."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / 'artifacts/core_revision'


def fmt(x):
    return f'{x:.3f}' if 0 < abs(x) < .005 else f'{x:.2f}'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clock = pd.read_csv(PAPER / 'artifacts/astra_revision/clock_path_cells.csv')
    choice = pd.read_csv(PAPER / 'artifacts/astra_revision/selection_consequence_cells.csv')
    np.testing.assert_allclose(clock.d_eval, clock.L34 / clock.L33 - 1, atol=1e-12)
    np.testing.assert_allclose(choice.excess_rel, (choice.L34_old-choice.L34_best)/choice.L34_best, atol=1e-12)
    periodic = clock[clock.operator.eq('PERIODIC')]
    np.testing.assert_allclose(periodic.d_eval, 0, atol=1e-12)
    c = clock.query("process == 'TEP' and operator == 'SWINGING_DOOR' and score == 'MAE'")
    s = choice.query("process == 'TEP' and operator == 'SWINGING_DOOR' and score == 'MAE'")
    assert len(c) == 864 and len(s) == 288
    assert not c.duplicated(['region','model','fold','fault','severity_k']).any()
    rows, selection = [], []
    for excluded in [0, *sorted(c.fault.unique())]:
        cc = c if excluded == 0 else c[c.fault.ne(excluded)]
        ss = s if excluded == 0 else s[s.fault.ne(excluded)]
        for (region, model), g in cc.groupby(['region','model']):
            assert len(g) == (72 if excluded == 0 else 60)
            rows.append(dict(region=region, model=model, excluded_fault=int(excluded), n=len(g),
                positive=int((g.d_eval > 0).sum()), median=100*g.d_eval.median(),
                q25=100*g.d_eval.quantile(.25), q75=100*g.d_eval.quantile(.75),
                minimum=100*g.d_eval.min(), maximum=100*g.d_eval.max()))
        for region, g in ss.groupby('region'):
            q = g[g.margin_qualified_change.eq(1)]
            selection.append(dict(region=region, excluded_fault=int(excluded), n=len(g),
                changed=int(g.changed.sum()), qualified=len(q),
                qualified_excess_below_1pct=int((q.excess_rel < .01).sum()),
                excess_median_all=100*g.excess_rel.median(),
                excess_max_all=100*g.excess_rel.max(),
                qualified_excess_min=100*q.excess_rel.min(), qualified_excess_max=100*q.excess_rel.max()))
    result, selections = pd.DataFrame(rows), pd.DataFrame(selection)
    result.to_csv(OUT / 'fault_exclusion_clock.csv', index=False)
    selections.to_csv(OUT / 'fault_exclusion_selection.csv', index=False)
    labels = {'PRESSURE_Q6':'Pressure, 6 min','PRESSURE_Q16':'Pressure, 16 min',
              'TEMPERATURE_Q6':'Temperature, 6 min','TEMPERATURE_Q16':'Temperature, 16 min'}
    models = {'RIDGE_ARX':'Ridge','PERSISTENCE':'Persistence','SHALLOW_HGB':'Tree'}
    lines = [r'\begin{tabular}{@{}llrrr@{}}', r'\toprule',
             r'Task & Model & All faults & Leave-one-fault-out range & Positive \\', r'\midrule']
    for region in labels:
        for model in models:
            g = result[result.region.eq(region) & result.model.eq(model)]
            a = g[g.excluded_fault.eq(0)].iloc[0]
            loo = g[g.excluded_fault.ne(0)]
            lines.append(f'{labels[region]} & {models[model]} & {a["median"]:.2f} & '
                         f'{fmt(loo["median"].min())} to {fmt(loo["median"].max())} & {int(a.positive)}/{int(a.n)} ' + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (PAPER / 'tables/generated/core_fault_exclusion.tex').write_text('\n'.join(lines)+'\n')
    checks = dict(source_clock_rows=len(clock), source_choice_rows=len(choice),
                  evaluation_ratios_recomputed=True, selection_excess_recomputed=True,
                  periodic_zero=True, all_six_fault_exclusions=True,
                  interpretation='Dependent design-cell summaries, not confidence intervals or refitted leave-fault-out models.')
    (OUT / 'e1_checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    print(result[result.model.eq('RIDGE_ARX')].to_string(index=False))
    print(selections[selections.excluded_fault.isin([0,14])].to_string(index=False))

    if (OUT/'fallback/cells.csv').exists():
        f=pd.read_csv(OUT/'fallback/cells.csv')
        t=pd.read_csv(OUT/'fallback/trajectory_losses.csv')
        keys=['region','fold','fault','operator','severity_k','fallback','score']
        aggregate=[]
        for key,g in t.groupby(keys):
            row=dict(zip(keys,key))
            assert len(g)==2 and g.n_test.nunique()==1
            for loss in ('L33','L34'):
                row[loss]=float(np.sqrt(np.mean(g[loss]**2))) if row['score']=='RMSE' else g[loss].mean()
            aggregate.append(row)
        paired=f.merge(pd.DataFrame(aggregate),on=keys,suffixes=('','_from_trajectories'),validate='one_to_one')
        for loss in ('L33','L34'):
            np.testing.assert_allclose(paired[loss],paired[loss+'_from_trajectories'],rtol=1e-12,atol=1e-12)
        summary=pd.read_csv(OUT/'fallback/summary.csv').query("operator == 'SWINGING_DOOR'")
        lines=[r'\begin{tabular}{@{}llrrrr@{}}',r'\toprule',
               r'Task & Score & ZOH median & Past-slope median & ZOH positive & Past-slope positive \\',r'\midrule']
        for region in labels:
            for score in ('MAE','RMSE'):
                g=summary[summary.region.eq(region)&summary.score.eq(score)].set_index('fallback')
                z,a=g.loc['hold'],g.loc['capped_slope']
                lines.append(f'{labels[region]} & {score} & {z.median_pct:.2f} & {a.median_pct:.2f} & '
                             f'{int(z.positive)}/72 & {int(a.positive)}/72 '+r'\\')
        lines += [r'\bottomrule',r'\end{tabular}']
        (PAPER/'tables/generated/core_fallback.tex').write_text('\n'.join(lines)+'\n')
        detail=[]
        for (region,fallback,score),g in t.query("operator == 'SWINGING_DOOR'").groupby(['region','fallback','score']):
            detail.append(dict(region=region,fallback=fallback,score=score,n=len(g),positive=int((g.d_eval>0).sum()),
                               median_pct=100*g.d_eval.median()))
        pd.DataFrame(detail).to_csv(OUT/'fallback/trajectory_summary.csv',index=False)
        checks['trajectory_losses_reaggregate_to_cells']=True
        (OUT/'e1_checks.json').write_text(json.dumps(checks,indent=2)+'\n')


if __name__ == '__main__':
    main()
