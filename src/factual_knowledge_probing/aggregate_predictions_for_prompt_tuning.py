import os
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str)
parser.add_argument('--dataset_type', type=str, choices=['test', 'train'])
parser.add_argument('--dataset_name', type=str, default='LAMA_TREx')
args = parser.parse_args()

data_path = os.path.join('data', args.dataset_name, 'train_relation_wise')
rel_ids = []
for fname in os.listdir(data_path):
    rel_id = fname.split('.')[0]
    if rel_id != 'all':
        rel_ids.append(rel_id)

aggregated_preds = []

for rel_id in rel_ids:
    with open(os.path.join(args.file_path, rel_id, f'pred_{args.dataset_type}_relation_wise_{rel_id}.json'), 'r') as fin:
        preds = json.load(fin)

    for pred in preds:
        aggregated_preds.append(pred)

# Save the aggregated predictions.
with open(os.path.join(args.file_path, f'pred_{args.dataset_name}_{args.dataset_type}.json'), 'w') as fout:
    json.dump(aggregated_preds, fout)