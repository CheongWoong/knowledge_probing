from collections import defaultdict
import argparse
import json
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    args = parser.parse_args()

    with open(f"data/{args.dataset_name}/train.json", 'r') as fin:
        f_train = json.load(fin)
    with open(f"data/{args.dataset_name}/test.json", 'r') as fin:
        f_test = json.load(fin)

    f_train_relation_wise = defaultdict(list)
    f_test_relation_wise = defaultdict(list)

    for example in f_train:
        f_train_relation_wise[example['rel_id']].append(example)
    for example in f_test:
        f_test_relation_wise[example['rel_id']].append(example)

    os.makedirs(f"data/{args.dataset_name}/train_relation_wise", exist_ok=True)
    for rel_id in f_train_relation_wise:
        with open(f"data/{args.dataset_name}/train_relation_wise/{rel_id}.json", 'w') as fout:
            json.dump(f_train_relation_wise[rel_id], fout)

    os.makedirs(f"data/{args.dataset_name}/test_relation_wise", exist_ok=True)
    for rel_id in f_test_relation_wise:
        with open(f"data/{args.dataset_name}/test_relation_wise/{rel_id}.json", 'w') as fout:
            json.dump(f_test_relation_wise[rel_id], fout)

    with open(f"data/{args.dataset_name}/all.json", 'r') as fin:
        f_all = json.load(fin)
    with open(f"data/{args.dataset_name}/train_relation_wise/all.json", 'w') as fout:
        json.dump(f_all, fout)
    with open(f"data/{args.dataset_name}/test_relation_wise/all.json", 'w') as fout:
        json.dump(f_all, fout)