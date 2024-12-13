from collections import defaultdict
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='analogy')
    args = parser.parse_args()

    with open(f'../data/{args.dataset_name}/MC_valid.json', 'r') as fin:
        train = json.load(fin)
    with open(f'../data/{args.dataset_name}/MC_test.json', 'r') as fin:
        test = json.load(fin)

    train_counts = defaultdict(int)
    test_counts = defaultdict(int)
    for sample in train:
        train_counts['all'] += 1
    for sample in test:
        test_counts['all'] += 1
    
    sorted_keys = sorted(list(test_counts.keys()), key=lambda x: int(x[1:]) if x[1:].isdigit() else 10000 if args.dataset_name=='LAMA_TREx' else x)

    print("\tTrain / %5s / %5s" % ('Test', 'All'))
    for key in sorted_keys:
        print(f"{key}:\t%5d / %5d / %5d" % (train_counts[key], test_counts[key], train_counts[key] + test_counts[key]))
    print()