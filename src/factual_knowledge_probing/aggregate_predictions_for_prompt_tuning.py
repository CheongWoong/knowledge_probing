import jsonlines
import json
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str)
parser.add_argument('--dataset_name', type=str, default='LAMA_TREx')
parser.add_argument('--dataset_type', type=str, choices=['test', 'train'])
args = parser.parse_args()

data_path = f'data/{args.dataset_name}/{args.dataset_type}_relation_wise'
rel_ids = []
for fname in os.listdir(data_path):
    rel_id = fname.split('.')[0]
    if rel_id != 'all':
        rel_ids.append(rel_id)

aggregated_preds = []

for rel_id in rel_ids:
    with jsonlines.open(os.path.join(args.model_name_or_path, rel_id, f'pred_{args.dataset_type}_relation_wise_{rel_id}.jsonl')) as fin:
        for pred in fin.iter():
            aggregated_preds.append(pred)

# Save the aggregated predictions.
with open(os.path.join(args.model_name_or_path, f'pred_{args.dataset_name}_{args.dataset_type}.jsonl'), 'w') as fout:
    for pred in aggregated_preds:
        json.dump(pred, fout)
        fout.write('\n')
