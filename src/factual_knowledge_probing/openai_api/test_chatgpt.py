import os
import argparse
from openai import OpenAI
import tiktoken

import json
import time
from tqdm.auto import tqdm
from nltk.corpus import stopwords


parser = argparse.ArgumentParser()
parser.add_argument('--target_model', type=str, default='gpt-3.5-turbo-0301')
parser.add_argument('--dataset_name', type=str, default='LAMA_TREx')
parser.add_argument('--dataset_type', type=str, default='test')
args = parser.parse_args()

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Make a list of stopwords to remove them in the output vocabulary
encoding = tiktoken.encoding_for_model(args.target_model)

stopword_list = stopwords.words("english")
# capitalized_stopword_list = []
# for word in stopword_list:
#     capitalized_stopword_list.append(word.capitalize())
# stopword_list = stopword_list + capitalized_stopword_list
stopword_list.append('The') # cwkang: we only add 'The' due to the maximum number limit of logit_bias in OpenAI API

stopword_ids = []
for stopword in stopword_list:
    token_ids = encoding.encode(' '+stopword)
    if len(token_ids) == 1:
        stopword_ids.append(token_ids[0])
    token_ids = encoding.encode(stopword)
    if len(token_ids) == 1:
        stopword_ids.append(token_ids[0])

logit_bias_remove_stopwords = {}
for stopword_id in stopword_ids:
    logit_bias_remove_stopwords[str(stopword_id)] = -100

# Load test data
with open(f'data/{args.dataset_name}/{args.dataset_type}.json') as fin:
    test_data = json.load(fin)

# Load valid example ids
with open(f'src/factual_knowledge_probing/openai_api/{args.dataset_name}/valid_uids.json', 'r') as fin:
    valid_uids = json.load(fin)

# Filter out invalid examples (whose answers are not in the ChatGPT's vocabulary)
uids = []
prompts = []

for example in tqdm(test_data):
    uid = example['uid']
    prompt = example['truncated_input']
    if uid in valid_uids:
        uids.append(uid)
        prompts.append(prompt)

# Run completions with API and store the results
raw_predictions = []
raw_predictions_remove_stopwords = []

batch_size = 1 # chatgpt does not support batched completion for now
for i in tqdm(range(0, len(prompts), batch_size)):
    uid_batch = uids[i:i+batch_size]
    prompt_batch = prompts[i:i+batch_size]
    messages = []
    for prompt in prompt_batch:
        messages.append({"role": "user", "content": prompt})

    while True:
        try:
            # responses = client.chat.completions.create(
			# 	model=args.target_model,
			# 	messages=messages,
			# 	max_tokens=1,
			# 	temperature=0,
            #     logprobs=True,
            #     top_logprobs=5,
			# )

            responses_remove_stopwords = client.chat.completions.create(
				model=args.target_model,
				messages=messages,
				max_tokens=1,
				temperature=0,
				logit_bias=logit_bias_remove_stopwords,
                logprobs=True,
                top_logprobs=5,
			)
            
            break
        except Exception as e:
            print('Error!', e)
            time.sleep(3)

    # for uid, response in zip(uid_batch, responses.choices):
    #     raw_predictions.append({"uid": uid, "response": response})
    # for uid, response_remove_stopwords in zip(uid_batch, responses_remove_stopwords.choices):
    #     raw_predictions_remove_stopwords.append({"uid": uid, "response": response_remove_stopwords})
    uid = uid_batch[0]
    logprobs_remove_stopwords = responses_remove_stopwords.choices[0].logprobs.content[0].top_logprobs
    top_5_tokens_remove_stopwords, top_5_logprobs_remove_stopwords = [], []
    for logprob in logprobs_remove_stopwords:
        top_5_tokens_remove_stopwords.append(logprob.token)
        top_5_logprobs_remove_stopwords.append(logprob.logprob)
    raw_predictions_remove_stopwords.append({"uid": uid, "top_5_tokens_remove_stopwords": top_5_tokens_remove_stopwords, "top_5_logprobs_remove_stopwords": top_5_logprobs_remove_stopwords})

# Write the results
out_path = os.path.join('results', args.target_model)
os.makedirs(out_path, exist_ok=True)

# with open(os.path.join(out_path, f'raw_pred_{args.dataset_name}_{args.dataset_type}.json'), 'w') as fout:
#     json.dump(raw_predictions, fout)

with open(os.path.join(out_path, f'raw_pred_{args.dataset_name}_{args.dataset_type}_remove_stopwords.json'), 'w') as fout:
    json.dump(raw_predictions_remove_stopwords, fout)