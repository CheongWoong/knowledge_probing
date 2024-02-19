from collections import defaultdict
import argparse
import json

import random
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--k', type=int, default=4)
    args = parser.parse_args()

    with open(f"data/{args.dataset_name}/train.json", 'r') as fin:
        f_train = json.load(fin)
    with open(f"data/{args.dataset_name}/test.json", 'r') as fin:
        f_test = json.load(fin)

    demonstrations = defaultdict(list)

    for example in f_train:
        rel = example['rel_id']
        demonstration = example['input'].replace('[MASK]', example['output'])
        demonstrations[rel].append(demonstration)

    random.seed(0)
    np.random.seed(0)
    
    for example in f_test:
        k = min(len(demonstrations[example['rel_id']]), args.k)
        if k >= 1:
            few_shot_examples = '\n'.join(random.sample(demonstrations[example['rel_id']], k=k))
        else:
            few_shot_examples = ''
        example['input'] = few_shot_examples + '\n' + example['input']
        example['truncated_input'] = few_shot_examples + '\n' + example['truncated_input']
        
    with open(f"data/{args.dataset_name}/test_{args.k}_shot.json", 'w') as fout:
        json.dump(f_test, fout)