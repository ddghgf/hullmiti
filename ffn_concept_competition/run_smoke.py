#!/usr/bin/env python3
from __future__ import annotations

import json, statistics
from pathlib import Path
import torch
from tqdm import tqdm

from ffn_concept_competition.controller import FCCConfig, FFNConceptCompetitionController
from ffn_concept_competition.eval_utils import generate_pope
from llava.eval.pope_baseline import compute_metrics
from pem.eval_utils import ROOT, add_fpr, efficiency, load_model, pope_rows


def delta(base, edited):
    norms=[float(torch.linalg.vector_norm(b['_logits']-a['_logits'])) for a,b in zip(base,edited)]
    margins=[b['yes_no_margin']-a['yes_no_margin'] for a,b in zip(base,edited)]
    return {'changed':sum(x>0 for x in norms),'mean_l2':statistics.fmean(norms),'max_l2':max(norms),
            'mean_margin_delta':statistics.fmean(margins),'top1_changes':sum(int(a['_logits'].argmax())!=int(b['_logits'].argmax()) for a,b in zip(base,edited))}


def main():
    out=ROOT/'outputs'/'ffn_concept_competition'; out.mkdir(parents=True,exist_ok=True)
    tokenizer,model,processor=load_model(); versions=[p._version for p in model.parameters()]
    data=pope_rows('random',10)
    base=[generate_pope(x,tokenizer,model,processor,capture=True) for x in tqdm(data,desc='FCC smoke baseline')]
    report={'baseline':{'metrics':add_fpr(compute_metrics(base)),'efficiency':efficiency(base)},'configs':{}}
    for name,config in (('strength0',FCCConfig(strength=0)),('fcc_s025_f005',FCCConfig(strength=.25,neuron_fraction=.05)),('fcc_s050_f005',FCCConfig(strength=.5,neuron_fraction=.05))):
        controller=FFNConceptCompetitionController(model,config)
        rows=[generate_pope(x,tokenizer,model,processor,controller,True) for x in tqdm(data,desc=f'FCC smoke {name}')]
        controller.close()
        report['configs'][name]={'config':config.to_dict(),'metrics':add_fpr(compute_metrics(rows)),'logit_delta':delta(base,rows),
            'prediction_changes':sum(a['prediction']!=b['prediction'] for a,b in zip(base,rows)),
            'incorrect_to_correct':sum(a['prediction']!=a['label'] and b['prediction']==b['label'] for a,b in zip(base,rows)),
            'correct_to_incorrect':sum(a['prediction']==a['label'] and b['prediction']!=b['label'] for a,b in zip(base,rows)),
            'efficiency':efficiency(rows),'stats':[x.get('fcc_stats',{}) for x in rows]}
    if versions!=[p._version for p in model.parameters()]: raise RuntimeError('FCC changed model parameters')
    report['parameters_unchanged']=True
    (out/'smoke_test.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__': main()
