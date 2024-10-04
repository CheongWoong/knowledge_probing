import os
import math
import json
import argparse
from copy import deepcopy
from collections import defaultdict
from nltk.corpus import stopwords

from tqdm.auto import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str)
parser.add_argument('--dataset_name', type=str, default='LAMA_TREx')
parser.add_argument('--dataset_type', type=str, default='test')
args = parser.parse_args()

stopword_list = stopwords.words('english')

with open(f'data/{args.dataset_name}/all.json') as fin:
    data = json.load(fin)

label_map = {}
subj_rel_map = {}
subj_rel_pair_gold_objs = defaultdict(set)
for example in data:
    uid = example['uid']
    subj = example['subj']
    rel = example['rel_id']
    label = example['output']

    label_map[uid] = label
    subj_rel_map[uid] = f'{subj}_{rel}'
    subj_rel_pair_gold_objs[f'{subj}_{rel}'].add(label.strip().lower())

# with open(os.path.join(args.file_path, f'raw_pred_{args.dataset_name}_{args.dataset_type}.json'), 'r') as fin:
#     raw_preds = json.load(fin)
with open(os.path.join(args.file_path, f'raw_pred_{args.dataset_name}_{args.dataset_type}_remove_stopwords.json'), 'r') as fin:
    raw_preds_remove_stopwords = json.load(fin)

predictions = []
for raw_pred_remove_stopwords in tqdm(raw_preds_remove_stopwords):
    uid = raw_pred_remove_stopwords['uid']

    top_k_tokens_remove_stopwords = raw_pred_remove_stopwords['top_k_tokens_remove_stopwords']
    top_k_logprobs_remove_stopwords = raw_pred_remove_stopwords['top_k_logprobs_remove_stopwords']
    top_k_probs_remove_stopwords = []
    for logprob in top_k_logprobs_remove_stopwords:
        top_k_probs_remove_stopwords.append(math.exp(logprob))

    label_text = label_map[uid].strip().lower()
    subj_rel = subj_rel_map[uid]
    subj_rel_gold_objs = deepcopy(subj_rel_pair_gold_objs[subj_rel])
    subj_rel_gold_objs.remove(label_text)
    preds_remove_stopwords = []
    for text in top_k_tokens_remove_stopwords:
        pred_remove_stopwords = text.strip().lower()
        if pred_remove_stopwords in subj_rel_gold_objs:
            continue
        elif pred_remove_stopwords in stopword_list: # cwkang: to handle stopwords in the output
            continue
        else:
            preds_remove_stopwords.append(pred_remove_stopwords)
    hits_1_remove_stopwords = (preds_remove_stopwords[0] == label_text)*1.0
    hits_10_remove_stopwords = 0.0
    for pred in preds_remove_stopwords[:10]:
        if pred == label_text:
            hits_10_remove_stopwords = 1.0

    prediction = {
        'uid': uid,
        'label_text': label_text,
        'top_k_text_remove_stopwords': top_k_tokens_remove_stopwords,
        'top_k_logprobs_remove_stopwords': top_k_logprobs_remove_stopwords,
        'top_k_probs_remove_stopwords': top_k_probs_remove_stopwords,
        'hits@1_remove_stopwords': hits_1_remove_stopwords,
        'hits@10_remove_stopwords': hits_10_remove_stopwords,
    }
    predictions.append(prediction)

with open(os.path.join(args.file_path, f'pred_{args.dataset_name}_{args.dataset_type}.jsonl'), 'w') as fout:
    for prediction in predictions:
        json.dump(prediction, fout)
        fout.write('\n')