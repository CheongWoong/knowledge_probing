import jsonlines
import json
import argparse
from collections import defaultdict
import numpy as np
from tqdm.auto import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--pred_file', type=str)
parser.add_argument('--reference_file', type=str)
args = parser.parse_args()

# Compute the uid-rel map for future use.
with open(args.reference_file, 'r') as fin:
    data = json.load(fin)

uid_rel_map = {}
for instance in data:
    uid = instance['uid']
    rel_id = instance['rel_id']
    uid_rel_map[uid] = rel_id

# Read the scores.
hits_1_total, hits_1_total_remove_stopwords, hits_1_total_gold_objs, hits_1_total_gold_objs_relation_wise = [], [], [], []
hits_1_relation_wise, hits_1_relation_wise_remove_stopwords, hits_1_relation_wise_gold_objs, hits_1_relation_wise_gold_objs_relation_wise = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
hits_10_total, hits_10_total_remove_stopwords, hits_10_total_gold_objs, hits_10_total_gold_objs_relation_wise = [], [], [], []
hits_10_relation_wise, hits_10_relation_wise_remove_stopwords, hits_10_relation_wise_gold_objs, hits_10_relation_wise_gold_objs_relation_wise = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
hits_100_total, hits_100_total_remove_stopwords, hits_100_total_gold_objs, hits_100_total_gold_objs_relation_wise = [], [], [], []
hits_100_relation_wise, hits_100_relation_wise_remove_stopwords, hits_100_relation_wise_gold_objs, hits_100_relation_wise_gold_objs_relation_wise = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
mrr_total, mrr_total_remove_stopwords, mrr_total_gold_objs, mrr_total_gold_objs_relation_wise = [], [], [], []
mrr_relation_wise, mrr_relation_wise_remove_stopwords, mrr_relation_wise_gold_objs, mrr_relation_wise_gold_objs_relation_wise = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)

with jsonlines.open(args.pred_file) as fin:
    for pred in tqdm(fin.iter()):
        uid = pred['uid']
        rel_id = uid_rel_map[uid]

        hits_1 = pred.get('hits@1', 0)
        hits_1_remove_stopwords = pred.get('hits@1_remove_stopwords', 0)
        hits_1_gold_objs = pred.get('hits@1_gold_objs', 0)
        hits_1_gold_objs_relation_wise = pred.get('hits@1_gold_objs_relation_wise', 0)
        hits_10 = pred.get('hits@10', 0)
        hits_10_remove_stopwords = pred.get('hits@10_remove_stopwords', 0)
        hits_10_gold_objs = pred.get('hits@10_gold_objs', 0)
        hits_10_gold_objs_relation_wise = pred.get('hits@10_gold_objs_relation_wise', 0)
        hits_100 = pred.get('hits@100', 0)
        hits_100_remove_stopwords = pred.get('hits@100_remove_stopwords', 0)
        hits_100_gold_objs = pred.get('hits@100_gold_objs', 0)
        hits_100_gold_objs_relation_wise = pred.get('hits@100_gold_objs_relation_wise', 0)
        mrr = pred.get('mrr', 0)
        mrr_remove_stopwords = pred.get('mrr_remove_stopwords', 0)
        mrr_gold_objs = pred.get('mrr_gold_objs', 0)
        mrr_gold_objs_relation_wise = pred.get('mrr_gold_objs_relation_wise', 0)

        hits_1_total.append(hits_1)
        hits_1_total_remove_stopwords.append(hits_1_remove_stopwords)
        hits_1_total_gold_objs.append(hits_1_gold_objs)
        hits_1_total_gold_objs_relation_wise.append(hits_1_gold_objs_relation_wise)
        hits_1_relation_wise[rel_id].append(hits_1)
        hits_1_relation_wise_remove_stopwords[rel_id].append(hits_1_remove_stopwords)
        hits_1_relation_wise_gold_objs[rel_id].append(hits_1_gold_objs)
        hits_1_relation_wise_gold_objs_relation_wise[rel_id].append(hits_1_gold_objs_relation_wise)
        
        hits_10_total.append(hits_10)
        hits_10_total_remove_stopwords.append(hits_10_remove_stopwords)
        hits_10_total_gold_objs.append(hits_10_gold_objs)
        hits_10_total_gold_objs_relation_wise.append(hits_10_gold_objs_relation_wise)
        hits_10_relation_wise[rel_id].append(hits_10)
        hits_10_relation_wise_remove_stopwords[rel_id].append(hits_10_remove_stopwords)
        hits_10_relation_wise_gold_objs[rel_id].append(hits_10_gold_objs)
        hits_10_relation_wise_gold_objs_relation_wise[rel_id].append(hits_10_gold_objs_relation_wise)

        hits_100_total.append(hits_100)
        hits_100_total_remove_stopwords.append(hits_100_remove_stopwords)
        hits_100_total_gold_objs.append(hits_100_gold_objs)
        hits_100_total_gold_objs_relation_wise.append(hits_100_gold_objs_relation_wise)
        hits_100_relation_wise[rel_id].append(hits_100)
        hits_100_relation_wise_remove_stopwords[rel_id].append(hits_100_remove_stopwords)
        hits_100_relation_wise_gold_objs[rel_id].append(hits_100_gold_objs)
        hits_100_relation_wise_gold_objs_relation_wise[rel_id].append(hits_100_gold_objs_relation_wise)

        mrr_total.append(mrr)
        mrr_total_remove_stopwords.append(mrr_remove_stopwords)
        mrr_total_gold_objs.append(mrr_gold_objs)
        mrr_total_gold_objs_relation_wise.append(mrr_gold_objs_relation_wise)
        mrr_relation_wise[rel_id].append(mrr)
        mrr_relation_wise_remove_stopwords[rel_id].append(mrr_remove_stopwords)
        mrr_relation_wise_gold_objs[rel_id].append(mrr_gold_objs)
        mrr_relation_wise_gold_objs_relation_wise[rel_id].append(mrr_gold_objs_relation_wise)

# Compute aggregate statistics (mean, standard deviation).
def mean_and_std(x):
    return np.mean(x), np.std(x)
result = {}
result['hits@1'] = f"%.2f +- %.2f" % (mean_and_std(hits_1_total))
result['hits@1_remove_stopwords'] = f"%.2f +- %.2f" % (mean_and_std(hits_1_total_remove_stopwords))
result['hits@1_gold_objs'] = f"%.2f +- %.2f" % (mean_and_std(hits_1_total_gold_objs))
result['hits@1_gold_objs_relation_wise'] = f"%.2f +- %.2f" % (mean_and_std(hits_1_total_gold_objs_relation_wise))

result['hits@10'] = f"%.2f +- %.2f" % (mean_and_std(hits_10_total))
result['hits@10_remove_stopwords'] = f"%.2f +- %.2f" % (mean_and_std(hits_10_total_remove_stopwords))
result['hits@10_gold_objs'] = f"%.2f +- %.2f" % (mean_and_std(hits_10_total_gold_objs))
result['hits@10_gold_objs_relation_wise'] = f"%.2f +- %.2f" % (mean_and_std(hits_10_total_gold_objs_relation_wise))

result['hits@100'] = f"%.2f +- %.2f" % (mean_and_std(hits_100_total))
result['hits@100_remove_stopwords'] = f"%.2f +- %.2f" % (mean_and_std(hits_100_total_remove_stopwords))
result['hits@100_gold_objs'] = f"%.2f +- %.2f" % (mean_and_std(hits_100_total_gold_objs))
result['hits@100_gold_objs_relation_wise'] = f"%.2f +- %.2f" % (mean_and_std(hits_100_total_gold_objs_relation_wise))

result['mrr'] = f"%.2f +- %.2f" % (mean_and_std(mrr_total))
result['mrr_remove_stopwords'] = f"%.2f +- %.2f" % (mean_and_std(mrr_total_remove_stopwords))
result['mrr_gold_objs'] = f"%.2f +- %.2f" % (mean_and_std(mrr_total_gold_objs))
result['mrr_gold_objs_relation_wise'] = f"%.2f +- %.2f" % (mean_and_std(mrr_total_gold_objs_relation_wise))
print(len(hits_1_total))

try:
    rel_ids = sorted(list(hits_1_relation_wise.keys()), key=lambda x: int(x[1:]))
except:
    rel_ids = sorted(list(hits_1_relation_wise.keys()))

for rel_id in rel_ids:
    print(rel_id, len(hits_1_relation_wise[rel_id]))
    result['hits@1_' + rel_id] = f"%.2f +- %.2f" % (mean_and_std(hits_1_relation_wise[rel_id]))
for rel_id in rel_ids:
    result['hits@1_remove_stopwords_' + rel_id] = f"%.2f +- %.2f" % (mean_and_std(hits_1_relation_wise_remove_stopwords[rel_id]))
for rel_id in rel_ids:
    result['hits@1_gold_objs_' + rel_id] = f"%.2f +- %.2f" % (mean_and_std(hits_1_relation_wise_gold_objs[rel_id]))
for rel_id in rel_ids:
    result['hits@1_gold_objs_relation_wise_' + rel_id] = f"%.2f +- %.2f" % (mean_and_std(hits_1_relation_wise_gold_objs_relation_wise[rel_id]))

for rel_id in rel_ids:
    result['hits@10_' + rel_id] = f"%.2f +- %.2f" % (mean_and_std(hits_10_relation_wise[rel_id]))
for rel_id in rel_ids:
    result['hits@10_remove_stopwords_' + rel_id] = f"%.2f +- %.2f" % (mean_and_std(hits_10_relation_wise_remove_stopwords[rel_id]))
for rel_id in rel_ids:
    result['hits@10_gold_objs_' + rel_id] = f"%.2f +- %.2f" % (mean_and_std(hits_10_relation_wise_gold_objs[rel_id]))
for rel_id in rel_ids:
    result['hits@10_gold_objs_relation_wise_' + rel_id] = f"%.2f +- %.2f" % (mean_and_std(hits_10_relation_wise_gold_objs_relation_wise[rel_id]))

for rel_id in rel_ids:
    result['hits@100_' + rel_id] = f"%.2f +- %.2f" % (mean_and_std(hits_100_relation_wise[rel_id]))
for rel_id in rel_ids:
    result['hits@100_remove_stopwords_' + rel_id] = f"%.2f +- %.2f" % (mean_and_std(hits_100_relation_wise_remove_stopwords[rel_id]))
for rel_id in rel_ids:
    result['hits@100_gold_objs_' + rel_id] = f"%.2f +- %.2f" % (mean_and_std(hits_100_relation_wise_gold_objs[rel_id]))
for rel_id in rel_ids:
    result['hits@100_gold_objs_relation_wise_' + rel_id] = f"%.2f +- %.2f" % (mean_and_std(hits_100_relation_wise_gold_objs_relation_wise[rel_id]))

for rel_id in rel_ids:
    result['mrr_' + rel_id] = f"%.2f +- %.2f" % (mean_and_std(mrr_relation_wise[rel_id]))
for rel_id in rel_ids:
    result['mrr_remove_stopwords_' + rel_id] = f"%.2f +- %.2f" % (mean_and_std(mrr_relation_wise_remove_stopwords[rel_id]))
for rel_id in rel_ids:
    result['mrr_gold_objs_' + rel_id] = f"%.2f +- %.2f" % (mean_and_std(mrr_relation_wise_gold_objs[rel_id]))
for rel_id in rel_ids:
    result['mrr_gold_objs_relation_wise_' + rel_id] = f"%.2f +- %.2f" % (mean_and_std(mrr_relation_wise_gold_objs_relation_wise[rel_id]))

# Save the score file.
with open(args.pred_file.replace('pred', 'score')[:-1], 'w') as fout:
    json.dump(result, fout, indent=4)
