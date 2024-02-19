import os
import argparse
import tiktoken

import json
from tqdm.auto import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='LAMA_TREx')
args = parser.parse_args()

# target_model = 'text-davinci-003'
# encoding = tiktoken.encoding_for_model(target_model)
target_model2 = 'gpt-3.5-turbo'
encoding2 = tiktoken.encoding_for_model(target_model2)

with open(f'data/{args.dataset_name}/all.json') as fin:
    data = json.load(fin)


valid_uids = []

count, count2 = 0, 0
for example in tqdm(data):
    uid = example['uid']
    prompt = example['truncated_input']
    obj = example['output']

    token_ids = encoding2.encode(obj)
    if len(token_ids) == 1:
        valid_uids.append(uid)
        count += 1
    else:
        count2 += 1

print(count, 'examples are valid among', count+count2, 'samples')

file_directory = os.path.dirname(__file__)
out_dir = os.path.join(file_directory, args.dataset_name)
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'valid_uids.json'), 'w') as fout:
    json.dump(valid_uids, fout)
