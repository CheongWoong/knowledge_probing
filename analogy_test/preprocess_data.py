from tqdm.auto import tqdm
import jsonlines
import json
import os
import argparse
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--valid_or_test',
        default='test'
    )
    parser.add_argument(
        '--model_names',
        nargs='+',
        default=['EleutherAI/gpt-neo-125m', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6b',
                 'bert-base-uncased', 'bert-large-uncased',
                 'roberta-base', 'roberta-large',
                 'albert-base-v1', 'albert-large-v1', 'albert-xlarge-v1',
                 'albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2',
                 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3-8B-Instruct',
                 ])
    args = parser.parse_args()

    stopword_list = stopwords.words('english')
    capitalized_stopword_list = []
    for word in stopword_list:
        capitalized_stopword_list.append(word.capitalize())
    stopword_list = stopword_list + capitalized_stopword_list

    tokenizers = []
    for model_name in args.model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizers.append(tokenizer)

    symbols = ['a', 'b', 'c', 'd', 'e']

    MC_prompts = []
    prompts = []
    is_valids = {}

    dataset_names = ['bats', 'google', 'sat', 'u2', 'u4']
    for dataset_name in dataset_names:
        data_path = f'../data/analogy_test_dataset/{dataset_name}/{args.valid_or_test}.jsonl'

        with jsonlines.open(data_path) as fin:
            for idx, sample in tqdm(enumerate(fin.iter())):
                # print(sample)

                stem = sample['stem']
                choices = sample['choice']
                answer = sample['answer']

                is_valid = True
                for choice in choices:
                    if not is_valid:
                        break
                    for entity in choice:
                        if not is_valid:
                            break
                        # if entity in stopword_list:
                        #     is_valid = False
                        #     break

                        if entity not in is_valids:
                            is_valid = True
                            for tokenizer in tokenizers:
                                input_ids = tokenizer.encode(' '+entity, add_special_tokens=False)
                                if len(input_ids) != 1:
                                    is_valid = False
                                    break
                            is_valids[entity] = is_valid
                        else:
                            is_valid = is_valids[entity]

                # Enable data filtering
                # if not is_valid:
                #     continue

                uid = f'{dataset_name}_{args.valid_or_test}_{idx}'
                sentences = [f'{choice[0]} is to {choice[1]}' for choice in choices]
                full_query = f'{stem[0]} is to {stem[1]} as'
                for symbol, sentence in zip(symbols, sentences):
                    full_query += f' ({symbol}) {sentence},'
                full_query = full_query[:-1] + '.'

                BASE_QUERY = f'{stem[0]} is to {stem[1]} as '
                vanilla_sentences = [f'{choice[0]} is to {choice[1]} .' for choice in choices]
                vanilla_queries = [BASE_QUERY + vanilla_sentence for vanilla_sentence in vanilla_sentences] 
                vanilla_sentences_without_period = [f'{choice[0]} is to {choice[1]}' for choice in choices]
                vanilla_queries_without_period = [BASE_QUERY + vanilla_sentence_without_period for vanilla_sentence_without_period in vanilla_sentences_without_period]

                masked_sentences = [f'{choice[0]} is to [MASK] .' for choice in choices]
                masked_queries = [BASE_QUERY + masked_sentence for masked_sentence in masked_sentences]
                truncated_sentences = [f'{choice[0]} is to' for choice in choices]
                truncated_queries = [BASE_QUERY + truncated_sentence for truncated_sentence in truncated_sentences]

                MC_prompts.append({
                    'uid': uid,
                    'query': stem,
                    'choice': choices,
                    'input': full_query,
                    'output': symbols[answer]
                })

                for j in range(len(choices)):
                    prompts.append({
                        'uid': f'{uid}_{j}',
                        'input': vanilla_queries[j],
                        'input_without_period': vanilla_queries_without_period[j],
                        'masked_input': masked_queries[j],
                        'truncated_input': truncated_queries[j],
                        'output': choices[j][1],
                    })

    out_path = '../data/analogy'
    os.makedirs(out_path, exist_ok=True)
    print(len(MC_prompts), len(prompts))

    with open(os.path.join(out_path, f'MC_{args.valid_or_test}.json'), 'w') as fout:
        json.dump(MC_prompts, fout)
    with open(os.path.join(out_path, f'{args.valid_or_test}.json'), 'w') as fout:
        json.dump(prompts, fout)