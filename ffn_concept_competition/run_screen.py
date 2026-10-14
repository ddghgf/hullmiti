#!/usr/bin/env python3
from __future__ import annotations

import csv, json, statistics
from tqdm import tqdm

from ffn_concept_competition.controller import FCCConfig, FFNConceptCompetitionController
from ffn_concept_competition.eval_utils import generate_chair, generate_pope
from llava.eval.pope_baseline import compute_metrics
from pem.eval_utils import (ROOT, SPLITS, add_fpr, chair_rows, efficiency, evaluate_chair,
                            load_model, pope_rows, transitions)


def stats(rows):
    values=[r.get('fcc_stats',{}).get('means',{}) for r in rows if r.get('fcc_stats',{}).get('calls')]
    keys=set().union(*(x.keys() for x in values)) if values else set()
    return {k:statistics.fmean(x[k] for x in values) for k in keys}


def main():
    out=ROOT/'outputs'/'ffn_concept_competition'; out.mkdir(parents=True,exist_ok=True)
    tokenizer,model,processor=load_model(); versions=[p._version for p in model.parameters()]
    pope_data={s:pope_rows(s,50) for s in SPLITS}; chair_data=chair_rows(100)
    base_pope={s:[generate_pope(x,tokenizer,model,processor) for x in tqdm(rows,desc=f'FCC baseline POPE {s}')] for s,rows in pope_data.items()}
    base_chair=[generate_chair(x,tokenizer,model,processor) for x in tqdm(chair_data,desc='FCC baseline CHAIR')]
    report={'protocol':{'pope':'100 per split, greedy, max_new_tokens=4','chair':'100 fixed images, beam=3, max_new_tokens=64'},
        'baseline':{'pope':{s:{'metrics':add_fpr(compute_metrics(rows)),'efficiency':efficiency(rows)} for s,rows in base_pope.items()},
                    'chair':{'metrics':evaluate_chair(base_chair,out/'chair_baseline_100.jsonl'),'efficiency':efficiency(base_chair)}},'experiments':{}}
    all_pope=[{**x,'method':'baseline','split':s} for s,rows in base_pope.items() for x in rows]
    all_chair=[{**x,'method':'baseline'} for x in base_chair]
    configs=(('fcc_s025_f005',FCCConfig(strength=.25,neuron_fraction=.05)),('fcc_s050_f005',FCCConfig(strength=.5,neuron_fraction=.05)))
    for name,config in configs:
        controller=FFNConceptCompetitionController(model,config)
        pope={s:[generate_pope(x,tokenizer,model,processor,controller) for x in tqdm(rows,desc=f'{name} POPE {s}')] for s,rows in pope_data.items()}
        item={'config':config.to_dict(),'pope':{s:{'metrics':add_fpr(compute_metrics(rows)),'transitions':transitions(base_pope[s],rows),
              'efficiency':efficiency(rows),'statistics':stats(rows)} for s,rows in pope.items()}}
        all_pope.extend({**x,'method':name,'split':s} for s,rows in pope.items() for x in rows)
        if name=='fcc_s050_f005':
            chair=[generate_chair(x,tokenizer,model,processor,controller) for x in tqdm(chair_data,desc=f'{name} CHAIR')]
            item['chair']={'metrics':evaluate_chair(chair,out/f'chair_{name}_100.jsonl'),'efficiency':efficiency(chair),
                'caption_changes':sum(a['caption']!=b['caption'] for a,b in zip(base_chair,chair)),'statistics':stats(chair)}
            all_chair.extend({**x,'method':name} for x in chair)
        controller.close(); report['experiments'][name]=item
    if versions!=[p._version for p in model.parameters()]: raise RuntimeError('FCC changed model parameters')
    report['parameters_unchanged']=True
    (out/'screening.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    with (out/'pope_predictions.jsonl').open('w') as f:
        for x in all_pope: f.write(json.dumps(x,ensure_ascii=False)+'\n')
    with (out/'chair_captions.jsonl').open('w') as f:
        for x in all_chair: f.write(json.dumps(x,ensure_ascii=False)+'\n')
    with (out/'pope_screening.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(('method','split','accuracy','f1','precision','recall','fpr','yes_ratio','flips','fixed','broken','mean_seconds'))
        entries=[('baseline',report['baseline']['pope'])]+[(n,x['pope']) for n,x in report['experiments'].items()]
        for n,item in entries:
            for s,x in item.items():
                m=x['metrics']; t=x.get('transitions',{})
                w.writerow((n,s,m['accuracy'],m['f1'],m['precision'],m['recall'],m['false_positive_rate'],m['yes_ratio'],t.get('prediction_changes',0),t.get('incorrect_to_correct',0),t.get('correct_to_incorrect',0),x['efficiency']['mean_seconds']))
    with (out/'chair_screening.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(('method','CHAIRs','CHAIRi','Recall','Len','ObjectCount','caption_changes','mean_seconds'))
        for n,x in [('baseline',report['baseline']['chair']),('fcc_s050_f005',report['experiments']['fcc_s050_f005']['chair'])]:
            m=x['metrics']; w.writerow((n,m['CHAIRs'],m['CHAIRi'],m['Recall'],m['Len'],m['ObjectCount'],x.get('caption_changes',0),x['efficiency']['mean_seconds']))
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__': main()
