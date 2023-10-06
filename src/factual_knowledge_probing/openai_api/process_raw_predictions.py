import os
import json
import argparse

from tqdm.auto import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str)
parser.add_argument('--dataset_name', type=str, default='LAMA_TREx')
parser.add_argument('--dataset_type', type=str, default='test')
args = parser.parse_args()

with open(os.path.join('data', args.dataset_name, 'all.json')) as fin:
    data = json.load(fin)

rel_map = {}
label_map = {}

for instance in data:
    uid = instance['uid']
    rel_id = instance['rel_id']
    label = instance['output']
    rel_map[uid] = rel_id
    label_map[uid] = label

with open(os.path.join(args.file_path, f'raw_pred_{args.dataset_name}_{args.dataset_type}.json'), 'r') as fin:
    raw_preds = json.load(fin)
with open(os.path.join(args.file_path, f'raw_pred_{args.dataset_name}_{args.dataset_type}_remove_stopwords.json'), 'r') as fin:
    raw_preds_remove_stopwords = json.load(fin)

with open(f'src/factual_knowledge_probing/openai_api/{args.dataset_name}/valid_uids.json', 'r') as fin:
    valid_uids = json.load(fin)

valid_predictions = []
predictions = []
for raw_pred, raw_pred_remove_stopwords in tqdm(zip(raw_preds, raw_preds_remove_stopwords)):
    assert raw_pred['uid'] == raw_pred_remove_stopwords['uid']

    uid = raw_pred['uid']
    if 'text' in raw_pred['response']:
        pred = raw_pred['response']['text'].strip().lower()
        pred_remove_stopwords = raw_pred_remove_stopwords['response']['text'].strip().lower()
    else:
        pred = raw_pred['response']['message']['content'].strip().lower()
        pred_remove_stopwords = raw_pred_remove_stopwords['response']['message']['content'].strip().lower()
    label_text = label_map[uid].strip().lower()
    
    hits_1 = (pred == label_text)*1.0
    hits_1_remove_stopwords = (pred_remove_stopwords == label_text)*1.0

    prediction = {
        'uid': uid,
        'label_text': label_text,
        'top_1_text': pred,
        'top_1_text_remove_stopwords': pred_remove_stopwords,
        'hits@1': hits_1,
        'hits@1_remove_stopwords': hits_1_remove_stopwords,
    }
    predictions.append(prediction)
    if uid in valid_uids:
        valid_predictions.append(prediction)

with open(os.path.join(args.file_path, f'pred_{args.dataset_name}_{args.dataset_type}.json'), 'w') as fout:
    json.dump(predictions, fout)
with open(os.path.join(args.file_path, f'pred_{args.dataset_name}_{args.dataset_type}_valid_only.json'), 'w') as fout:
    json.dump(valid_predictions, fout)