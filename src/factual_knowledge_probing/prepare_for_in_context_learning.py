from collections import defaultdict
import argparse
import json
import os

import random
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()

    with open(os.path.join(args.data_path, 'train.json'), 'r') as fin:
        f_train = json.load(fin)
    with open(os.path.join(args.data_path, 'test.json'), 'r') as fin:
        f_test = json.load(fin)

    demonstrations = defaultdict(list)

    for example in f_train:
        rel = example['rel_id']
        demonstration = example['input'].replace('[MASK]', example['output'])
        demonstrations[rel].append(demonstration)

    random.seed(0)
    np.random.seed(0)
    
    for example in f_test:
        few_shot_examples = '\n'.join(random.sample(demonstrations[example['rel_id']], k=4))
        example['input'] = few_shot_examples + '\n' + example['input']
        example['truncated_input'] = few_shot_examples + '\n' + example['truncated_input']
        
    with open(os.path.join(args.data_path, 'test_4_shot.json'), 'w') as fout:
        json.dump(f_test, fout)