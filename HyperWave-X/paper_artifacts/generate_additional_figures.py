"""Plot existing ablation, sensitivity and representative episode evidence."""
import csv
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_support import HERE, prepare, save
sys.path.insert(0, str(HERE.parent))
from plot_interpretability_trace import load_episode, render

LANGUAGE = 'en'
RESULTS = HERE.parent / 'results'
METRICS = [('Task_Completion_Ratio','Task completion ratio'),
 ('Capacity_Completion_Ratio','Capacity completion ratio'),
 ('Synchronization_Success_Rate','Synchronization success rate'),
 ('Missing_Member_Rate','Missing-member rate'),
 ('Mean_Active_Link_Ratio','Active-link ratio')]

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def main():
    prepare(LANGUAGE)
    plt.rcParams.update({'font.size':9, 'axes.spines.top':False,'axes.spines.right':False})
    data = rows(RESULTS/'ablation_results'/'ablation_reviewer_metrics_summary.csv')
    variants = [('hyperwave_x','Full HyperWave-X'),('no_capability','No capability'),('no_wave','No wave'),('complete_topology','Complete topology')]
    actual = {r['Method'] for r in data}
    if 'hyperwave_x' not in actual:
        full = next((x for x in actual if x.lower() in ('full','hyperwave-x')),None)
        if full is None:
            raise ValueError(f'Full comparison missing from ablation summary: {actual}')
        variants[0]=(full,'Full HyperWave-X')
    fig,axes=plt.subplots(2,len(METRICS),figsize=(17,7),layout='constrained')
    for i,scenario in enumerate(['dynamic_overload','scale_20']):
        for ax,(metric,label) in zip(axes[i],METRICS):
            for j,(variant,name) in enumerate(variants):
                matches=[r for r in data if r['Scenario']=='lbf_'+scenario+'-v0' and r['Method']==variant and r['Metric']==metric]
                if len(matches)!=1: raise ValueError((scenario,variant,metric,len(matches)))
                r=matches[0]
                ax.errorbar(j,float(r['Mean']),yerr=float(r['CI95_Half_Width']),fmt='o',capsize=3)
            ax.set_title(scenario);ax.set_ylabel(label)
            ax.set_xticks(range(4),[v[1] for v in variants],rotation=55,ha='right');ax.grid(alpha=.2)
    save(fig,'Figure_9_Ablation_Study',LANGUAGE)
    data=rows(RESULTS/'sensitivity-results'/'performance_summary.csv')
    for factor,settings,xlabel,stem in [
        ('K_w',['wave_iterations_1','wave_iterations_2','wave_iterations_3'],'Wave iterations','Figure_10_Wave_Iterations'),
        ('Wave_Damping',['wave_damping_0p5','wave_iterations_2','wave_damping_0p95'],'Wave damping','Figure_11_Wave_Damping')]:
        fig,axes=plt.subplots(1,len(METRICS),figsize=(16,3.4),layout='constrained')
        for ax,(metric,label) in zip(axes,METRICS):
            selected=[]
            for setting in settings:
                matches=[r for r in data if r['Setting']==setting and r['Metric']==metric]
                if len(matches)!=1: raise ValueError((setting,metric,len(matches)))
                selected.append(matches[0])
            ax.errorbar([float(r[factor]) for r in selected],[float(r['Mean']) for r in selected],
                yerr=[float(r['CI95_Half_Width']) for r in selected],fmt='o-',capsize=3)
            ax.set_xlabel(xlabel);ax.set_ylabel(label);ax.set_title('consensus_wide');ax.grid(alpha=.2)
        save(fig,stem,LANGUAGE)
    trace=RESULTS/'PROBES_ORGANIZED_V1/hyperwave_x/lbf_probe_complementary-v0/seed_20/run1/episode_trace.jsonl'
    # Reuse the original trace renderer; redirect only its save and close operations.
    original_save=plt.Figure.savefig; original_close=plt.close
    captured=[]
    try:
        plt.Figure.savefig=lambda figure,*args,**kwargs: captured.append(figure)
        plt.close=lambda *args,**kwargs: None
        render(load_episode(trace,1),HERE/'unused.png')
    finally:
        plt.Figure.savefig=original_save;plt.close=original_close
    save(captured[0],'Figure_8_Representative_Episode',LANGUAGE)

if __name__=='__main__':
    main()
