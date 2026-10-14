from __future__ import annotations

import time
import torch

from llava.constants import IMAGE_TOKEN_INDEX
from llava.eval.pope_baseline import YesNoStoppingCriteria, build_prompt, pope_prediction
from llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token
from pem.eval_utils import ROOT, chair_prompt, margin, prepare_image


@torch.inference_mode()
def generate_pope(row, tokenizer, model, processor, controller=None, capture=False):
    image = prepare_image(ROOT/'DATA'/'val2014'/row['image'], processor, model)
    ids = tokenizer_image_token(build_prompt(row['text']), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    if controller: controller.begin_sample(ids)
    torch.cuda.reset_peak_memory_stats(model.device); torch.cuda.synchronize(); started=time.perf_counter()
    output=model.generate(ids,images=image,do_sample=False,num_beams=1,max_new_tokens=4,use_cache=True,
        stopping_criteria=[YesNoStoppingCriteria(tokenizer,ids.shape[1])],return_dict_in_generate=True,output_scores=capture)
    torch.cuda.synchronize(); elapsed=time.perf_counter()-started
    raw=tokenizer.decode(output.sequences[0,ids.shape[1]:],skip_special_tokens=True).strip()
    result={'source_index':row['source_index'],'image':row['image'],'question':row['text'],'label':row['label'].lower().strip(),
        'prediction':pope_prediction(raw),'raw_output':raw,'seconds':elapsed,
        'peak_gpu_memory_bytes':int(torch.cuda.max_memory_allocated(model.device))}
    if controller: result[controller.stats_key]=controller.summary()
    if capture:
        result['_logits']=output.scores[0][0].detach().float().cpu(); result['yes_no_margin']=margin(result['_logits'],tokenizer)
    return result


@torch.inference_mode()
def generate_chair(row, tokenizer, model, processor, controller=None, capture=False):
    image=prepare_image(row['image_path'],processor,model)
    ids=tokenizer_image_token(chair_prompt(row['question']),tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).to(model.device)
    if controller: controller.begin_sample(ids)
    torch.cuda.reset_peak_memory_stats(model.device); torch.cuda.synchronize(); started=time.perf_counter()
    output=model.generate(ids,images=image,attention_mask=torch.ones_like(ids),do_sample=False,temperature=0.0,
        top_p=None,top_k=None,num_beams=3,max_new_tokens=64,use_cache=True,
        stopping_criteria=[KeywordsStoppingCriteria(['###'],tokenizer,ids)],return_dict_in_generate=True,output_scores=capture)
    torch.cuda.synchronize(); elapsed=time.perf_counter()-started
    caption=tokenizer.decode(output.sequences[0,ids.shape[1]:],skip_special_tokens=True)
    if '###' in caption: caption=caption.split('###',1)[0]
    result={'image_id':row['image_id'],'caption':caption.strip(),'seconds':elapsed,
        'peak_gpu_memory_bytes':int(torch.cuda.max_memory_allocated(model.device))}
    if controller: result[controller.stats_key]=controller.summary()
    if capture: result['_logits']=output.scores[0][0].detach().float().cpu()
    return result
