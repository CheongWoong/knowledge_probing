# Factual Knowledge Probing with Autoregressive Language Models


## Installation

### Set up a Conda Environment
This setup script creates an environment named 'factual_knowledge_probing'.
```
bash scripts/installation/setup_conda.sh
```

### Download the LAMA TREx dataset
The original dataset is saved in 'data/original_LAMA'.  
The preprocessed dataset is saved in 'data/LAMA_TREx'.
```
bash scripts/installation/download_LAMA.sh
bash scripts/installation/preprocess_LAMA_TREx.sh
```

Check the number of samples for each relation.
```
bash scripts/installation/check_number_of_samples_LAMA_TREx.sh
```

### Download the Pretrained Models
The pretrained models (e.g. 'gpt_j_6B') are saved in 'results/{model_name}'.
```
bash scripts/installation/download_pretrained_models.sh
```


## Evaluation

### Test
The prediction file (e.g. 'pred_LAMA_TREX_test.json') is saved in '{model_path}'.
```
# Zero-shot test (optional: deepspeed zero-3)
# model_path: ['results/gpt_neo_125M', 'results/gpt_j_6B', ...]
# dataset_type: ['test', 'train', ...]
bash scripts/test/run_zeroshot.sh {model_path} {dataset_type}

# Test finetuned models
bash scripts/test/run_finetuned.sh {model_path} {dataset_type}
```

### Compute Score
This evaluation script computes score and saves the results in 'score_factual_probing_test.json'.
```
# prediction_file: ['results/gpt_neo_125M/pred_LAMA_TREx_test.json', ...]
bash scripts/test/compute_score.sh {prediction_file}
```


## Training

### Finetuning
The finetuned models and prediction files are saved in 'results/{model_name}_TREx'.
```
bash scripts/factual_knowledge_probing/finetuning/run_neo_125M.sh
bash scripts/factual_knowledge_probing/finetuning/run_j_6B_ds_zero3.sh
```

### Prompt Tuning
To be released soon.

### In-context Learning
The following script generates the test data with demonstrations (few-shot prompts).
```
bash scripts/factual_knowledge_probing/generate_few_shot_prompts.sh
```

## DeepSpeed Issues
We tried disabling P2P to resolve the hanging issues of DeepSpeed.
```
export nccl_p2p_disable=1
```
