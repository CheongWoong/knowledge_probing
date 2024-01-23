import os
from tqdm.auto import tqdm
from collections import defaultdict
from copy import deepcopy
import json

from scipy.special import softmax
import numpy as np
from nltk.corpus import stopwords

import torch


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    # return logits.argmax(dim=-1)
    ##### cw: modification for factual knowledge probing
    label_idx = torch.argmax((labels[:, 1:] >= 0)*1, dim=-1) ## find the index of obj in the (subj-rel-obj) triple.
    mask = torch.zeros(labels[:, 1:].shape, device=labels.device).scatter(1, label_idx.unsqueeze(1), 1.0) > 0.5
    logits = logits[:, -(mask.shape[1]+1):] ## cw: to match the shape (the input length is expanded when using prompt tuning methods)
    logits = logits[:, :-1][mask] ## get the logits at the label (obj) index.
    # return logits.detach().cpu()
    return logits ## keep logits on gpu for errors in the multi-gpu setting.
    #####

def get_masks(tokenizer, f_all):
    # generate the stopword mask to restrict candidate sets by removing stopwords.
    stopword_list = stopwords.words("english")
    stopword_ids = []
    for stopword in stopword_list:
        token_ids = tokenizer.encode(' '+stopword)
        if len(token_ids) == 1:
            stopword_ids.append(token_ids[0])
    stopword_mask = torch.tensor(stopword_ids, dtype=torch.int32)

    # generate the gold object mask to restrict candidate sets.
    gold_obj_ids = set()
    gold_obj_relation_wise_ids = defaultdict(set)
    subj_rel_pair_gold_obj_ids = defaultdict(set)

    for example in f_all:
        subj = example['subj']
        rel = example['rel_id']
        obj = example['output']
        obj_id = tokenizer.encode(' '+obj)[0]
        gold_obj_relation_wise_ids[rel].add(obj_id)
        subj_rel_pair_gold_obj_ids[f'{subj}_{rel}'].add(obj_id)
        gold_obj_ids.add(obj_id)

    ## compute negated ids (== words that are not gold objects)
    gold_obj_mask = [i for i in range(tokenizer.vocab_size)]
    gold_obj_relation_wise_mask = {}

    for gold_obj_id in gold_obj_ids:
        if gold_obj_id in gold_obj_mask:
            gold_obj_mask.remove(gold_obj_id)
    for rel in gold_obj_relation_wise_ids:
        gold_obj_relation_wise_mask[rel] = [i for i in range(tokenizer.vocab_size)]
        for gold_obj_id in gold_obj_relation_wise_ids[rel]:
            gold_obj_relation_wise_mask[rel].remove(gold_obj_id)

    ## set => list
    for key in subj_rel_pair_gold_obj_ids:
        subj_rel_pair_gold_obj_ids[key] = list(subj_rel_pair_gold_obj_ids[key])

    return stopword_mask, gold_obj_mask, gold_obj_relation_wise_mask, subj_rel_pair_gold_obj_ids

def postprocess_single_prediction(logits, logits_for_hits_1, probs, tokenizer, label_id, label_text):
    results = {}

    # compute top 100 predictions
    sorted_idx = np.argsort(logits)[::-1]
    top_100_idx = sorted_idx[:100]
    results["top_100_text"] = [tokenizer.decode(token_id).strip() for token_id in top_100_idx]
    results["top_100_logits"] = logits[top_100_idx].tolist()
    results["top_100_probs"] = probs[top_100_idx].tolist()
    # compute mrr
    results["mrr"] = 1/(np.where(sorted_idx == label_id)[0][0]+1)
    # compute hits@1
    top_1_idx = np.argsort(logits_for_hits_1)[::-1][0]
    top_1_text = tokenizer.decode(top_1_idx).strip().lower()
    results["hits@1"] = 1.0 if top_1_text == label_text else 0.0

    return results

def postprocess_predictions(predictions, label_ids, validation_dataset, validation_file_path, output_dir, tokenizer):
    # get the masks to restrict output candidate sets.
    with open(os.path.join(os.path.dirname(validation_file_path), 'all.json'), 'r') as fin:
        f_all = json.load(fin)

    stopword_mask, gold_obj_mask, gold_obj_relation_wise_mask, subj_rel_pair_gold_obj_ids = get_masks(tokenizer, f_all)
    
    label_ids = label_ids[:, 1:]
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    label_texts = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # post-process the predictions for evaluation and save.
    print("Processing output predictions...")
    predictions_output = []
    for idx, example in tqdm(enumerate(validation_dataset)):
        ## 1. default (no restriction)
        logits = predictions[idx]
        ## 2. 1 + remove stopwords
        logits_remove_stopwords = logits.copy()
        logits_remove_stopwords[stopword_mask] = -10000.
        ## 3. 1 + restrict candidates to the set of gold objects in the whole dataset
        logits_gold_objs = logits.copy()
        logits_gold_objs[gold_obj_mask] = -10000.
        ## 4. 1 + restrict candidates to the set of gold objects with the same relation
        logits_gold_objs_relation_wise = logits.copy()
        logits_gold_objs_relation_wise[gold_obj_relation_wise_mask[example['rel_id']]] = -10000.

        probs = softmax(logits)
        label_id = label_ids[idx]
        label_id = label_id[label_id != tokenizer.pad_token_id][0] ## we only consider a single-token (the first generated token) evaluation
        label_text = label_texts[idx].strip().lower()

        ## When computing hits@1, remove other gold objects for the given subj-rel pair.
        subj_rel_pair_gold_obj_mask = deepcopy(subj_rel_pair_gold_obj_ids[example['subj']+'_'+example['rel_id']])
        obj = example['output']
        obj_id = tokenizer.encode(' '+obj)[0]
        subj_rel_pair_gold_obj_mask.remove(obj_id)

        logits_for_hits_1 = logits.copy()
        logits_for_hits_1_remove_stopwords = logits_remove_stopwords.copy()
        logits_for_hits_1_gold_objs = logits_gold_objs.copy()
        logits_for_hits_1_gold_objs_relation_wise = logits_gold_objs_relation_wise.copy()

        logits_for_hits_1[subj_rel_pair_gold_obj_mask] = -10000.
        logits_for_hits_1_remove_stopwords[subj_rel_pair_gold_obj_mask] = -10000.
        logits_for_hits_1_gold_objs[subj_rel_pair_gold_obj_mask] = -10000.
        logits_for_hits_1_gold_objs_relation_wise[subj_rel_pair_gold_obj_mask] = -10000.

        ### Compute the results (top 100 predictions, MRR, hits@1)
        postprocessed_results = postprocess_single_prediction(logits, logits_for_hits_1, probs, tokenizer, label_id, label_text)
        postprocessed_results_remove_stopwords = postprocess_single_prediction(logits_remove_stopwords, logits_for_hits_1_remove_stopwords, probs, tokenizer, label_id, label_text)
        postprocessed_results_gold_objs = postprocess_single_prediction(logits_gold_objs, logits_for_hits_1_gold_objs, probs, tokenizer, label_id, label_text)
        postprocessed_results_gold_objs_relation_wise = postprocess_single_prediction(logits_gold_objs_relation_wise, logits_for_hits_1_gold_objs_relation_wise, probs, tokenizer, label_id, label_text)

        postprocessed_results_aggregated = {
            "uid": example["uid"],
            "label_text": label_text,
        }

        for key in postprocessed_results:
            postprocessed_results_aggregated[key] = postprocessed_results[key]
        for key in postprocessed_results_remove_stopwords:
            postprocessed_results_aggregated[f"{key}_remove_stopwords"] = postprocessed_results_remove_stopwords[key]
        for key in postprocessed_results_gold_objs:
            postprocessed_results_aggregated[f"{key}_gold_objs"] = postprocessed_results_gold_objs[key]
        for key in postprocessed_results_gold_objs_relation_wise:
            postprocessed_results_aggregated[f"{key}_gold_objs_relation_wise"] = postprocessed_results_gold_objs_relation_wise[key]
    
        predictions_output.append(postprocessed_results_aggregated)

    basename = os.path.basename(validation_file_path)
    dataset_name = os.path.basename(os.path.dirname(validation_file_path))
    with open(os.path.join(output_dir, f"pred_{dataset_name}_{basename}.json"), "w") as fout:
        json.dump(predictions_output, fout)
