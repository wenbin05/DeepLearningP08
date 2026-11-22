"""Validate the complete matrix with an explicitly separate full-FT memory protocol."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.run_final import read_protocol,matrix,run_name,check_training,check_evaluation,require_artifacts
from scripts.check_final_results import rescore
from src.aggregate_results import aggregate
from src.comparison import validate_comparison
from src.train import write_json

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--main-results',default='results/cpu_placement_v1')
parser.add_argument('--full-results',default='results/cpu_full_ft_batch1_v1')
parser.add_argument('--output-dir')
args=parser.parse_args()
repo=Path(__file__).resolve().parents[2]
main=repo/args.main_results
full=repo/args.full_results
p=read_protocol(repo/'experiments/cpu_placement_protocol.json')
f=read_protocol(repo/'experiments/cpu_full_ft_protocol.json')
allowed={'study','train_batch_size','gradient_accumulation_steps'}
assert {k:v for k,v in p.items() if k not in allowed}=={k:v for k,v in f.items() if k not in allowed}
assert p['train_batch_size']*p['gradient_accumulation_steps']==f['train_batch_size']*f['gradient_accumulation_steps']==32
paths={'validation':[],'test':[]};problems=[]
for mode,rank,projections in matrix('all'):
    root,protocol=(full,f) if mode=='full_ft' else (main,p)
    for seed in protocol['seeds']:
        directory=root/run_name(mode,rank,projections,seed)
        try:
            path=directory/'validation_results.json'
            training=json.loads(path.read_text())
            check_training(training,protocol,mode,rank,projections,seed)
            require_artifacts(directory)
            if json.loads((directory/'run_config.json').read_text())!=training['manifest']:
                raise ValueError('Standalone run manifest differs from result')
            rescore(directory/'validation_predictions.jsonl',training)
            paths['validation'].append(path)
            for dataset in ('cnn_dailymail','xsum'):
                candidates=list((directory/'evaluations').glob(f'{dataset}_*/test_results.json'))
                if len(candidates)!=1:raise ValueError(f'Expected one {dataset} test result')
                result=json.loads(candidates[0].read_text())
                check_evaluation(result,training,protocol,dataset)
                rescore(candidates[0].with_name('predictions.jsonl'),result)
                paths['test'].append(candidates[0])
        except (OSError,ValueError,KeyError,TypeError) as exc:
            problems.append({'run':str(directory),'error':str(exc)})
summary={}
if not problems:
    for phase,files in paths.items():
        try:
            rows=aggregate(files,expected_seeds=p['seeds'])
            for dataset in {r['dataset'] for r in rows}:
                non_full=[r for r in rows if r['dataset']==dataset and r['placement']!='full_ft']
                ft=[r for r in rows if r['dataset']==dataset and r['placement']=='full_ft']
                validate_comparison(non_full)
                validate_comparison(ft)
                # Verify shared data, model, decoding and metric provenance across the
                # explicitly different microbatch protocols; never rewrite saved metadata.
                a,b=non_full[0]['protocol'],ft[0]['protocol']
                for key in ('generation','metric_protocol','reference_protocol','metric_revision','model_config','evaluation'):
                    if a.get(key)!=b.get(key):raise ValueError(f'Full FT differs on shared {key}')
                if non_full[0]['placement']=='zero_shot':a=next(r['protocol'] for r in non_full if r['placement']!='zero_shot')
                if a['data']!=b['data']:raise ValueError('Full FT differs on data membership')
            summary[phase]=rows
        except (ValueError,KeyError,TypeError) as exc:problems.append({'phase':phase,'error':str(exc)})
output=repo/args.output_dir if args.output_dir else main/'all_checks'
report={'expected_runs':39,'experiment_matrix_complete':not problems,'rouge_recomputed':not problems,
        'main_study':str(main),'full_ft_study':str(full),
        'full_ft_microbatch_exception':{'main_microbatch':2,'full_ft_microbatch':1,'effective_batch_both':32,
        'interpretation':'Full FT is a memory-adjusted reference baseline; microbatching is not identical to the other configurations.'},
        'problems':problems}
write_json(output/'readiness.json',report)
if not problems:
    for phase,rows in summary.items():
        write_json(output/f'{phase}_summary.json',rows)
        write_json(output/f'{phase}_summary_non_full_ft.json',[r for r in rows if r['placement']!='full_ft'])
        write_json(output/f'{phase}_summary_full_ft.json',[r for r in rows if r['placement']=='full_ft'])
print(json.dumps(report,indent=2))
raise SystemExit(0 if not problems else 1)
