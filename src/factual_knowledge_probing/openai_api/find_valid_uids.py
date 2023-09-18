import os
import tiktoken

import json
from tqdm.auto import tqdm



# target_model = 'text-davinci-003'
# encoding = tiktoken.encoding_for_model(target_model)
target_model2 = 'gpt-3.5-turbo'
encoding2 = tiktoken.encoding_for_model(target_model2)

with open('data/LAMA_TREx/test.json') as fin:
    test_data = json.load(fin)


valid_uids = []

count, count2 = 0, 0
for example in tqdm(test_data):
    uid = example['uid']
    prompt = example['truncated_input']
    obj = example['output']

    token_ids = encoding2.encode(obj)
    if len(token_ids) == 1:
        valid_uids.append(uid)
        count += 1
    else:
        count2 += 1

print(count, count2)

file_directory = os.path.dirname(__file__)
with open(os.path.join(file_directory, 'valid_uids.json'), 'w') as fout:
    json.dump(valid_uids, fout)
