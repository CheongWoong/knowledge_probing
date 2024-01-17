# Factual Knowledge Probing with Autoregressive Language Models
This is a repository for factual knowledge probing, which is part of the EMNLP 2023 Findings paper "[Impact of Co-occurrence on Factual Knowledge of Large Language Models](https://aclanthology.org/2023.findings-emnlp.518.pdf)".  

<p align="center">
<img src="https://github.com/CheongWoong/cheongwoong.github.io/blob/master/assets/img/impact_of_cooccurrence/factual_knowledge_probing_procedure.png"></img>
<br><br><br>
<img src="https://github.com/CheongWoong/cheongwoong.github.io/blob/master/assets/img/impact_of_cooccurrence/factual_knowledge_probing_metrics.png"></img>
</p>


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

### Download the Pretrained Models (Optional)
The pretrained models (e.g. 'gpt_j_6B') are saved in 'results/{model_name}'.
```
bash scripts/installation/download_pretrained_models.sh
```


## Evaluation

### Test
The prediction file (e.g. 'pred_{dataset_name}_test.json') is saved in 'results/{model_name_or_path}_{dataset_name}'.
```
# Zero-shot test
# model_name_or_path: ['results/gpt_neo_125M', 'results/gpt_j_6B', 'meta-llama/Llama-2-7b-chat-hf', ...]
# dataset_type: ['test', 'train', ...]
bash scripts/factual_knowledge_probing/test/test_zeroshot.sh {model_type} {model_name_or_path} {dataset_name} {dataset_type}
# Optional: run with deepspeed ZeRO-3
bash scripts/factual_knowledge_probing/test/test_zeroshot_ds_zero3.sh {model_type} {model_name_or_path} {dataset_name} {dataset_type}

# Test finetuned models
bash scripts/factual_knowledge_probing/test/test_finetuned.sh {model_type} {model_name_or_path} {dataset_name} {dataset_type}

# Test prompt-tuned models (Note that this script tests the model for every relation iteratively.)
bash scripts/factual_knowledge_probing/test/test_prompt_tuned.sh {model_type} {model_name_or_path} {dataset_name} {dataset_type}
# Aggregate predictions of prompt-tuned models
bash scripts/factual_knowledge_probing/test/aggregate_predictions_for_prompt_tuning.sh {model_name_or_path} {dataset_name} {dataset_type}
```

### Compute Score
This evaluation script computes score and saves the results in 'score_factual_probing_test.json'.
```
# prediction_file: ['results/gpt_neo_125M_LAMA_TREx_finetuning/pred_LAMA_TREx_test.json', ...]
bash scripts/test/compute_score.sh {prediction_file} {dataset_name}
```


## Training

### Finetuning
Run the following script to finetune the model.  
The finetuned models and prediction files are saved in 'results/{model_name_or_path}_{dataset_name}_finetuning'.
```
bash scripts/factual_knowledge_probing/finetuning/finetuning.sh {model_name_or_path}
```

### Prompt Tuning
Run the following script to train the model with prompt tuning.  
The finetuned models and prediction files are saved in 'results/{model_name_or_path}_{dataset_name}_prompt_tuning'.  
Note that this script runs prompt tuning for every relation iteratively.
```
bash scripts/factual_knowledge_probing/prompt_tuning/prompt_tuning.sh {model_name_or_path}
```

### In-context Learning
The following script generates the test data with demonstrations (few-shot prompts).  
Then, you can test the models with few-shot prompts.
```
bash scripts/factual_knowledge_probing/in_context_learning/prepare_for_in_context_learning.sh
bash scripts/factual_knowledge_probing/test/test_zeroshot.sh {model_name_or_path} test_4_shot
```
