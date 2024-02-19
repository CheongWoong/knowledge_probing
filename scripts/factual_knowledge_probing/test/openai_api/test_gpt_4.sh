target_model='gpt-4-0125-preview'
dataset_name=$1
dataset_type='test'

python -m src.factual_knowledge_probing.openai_api.test_chatgpt --target_model $target_model --dataset_name $dataset_name --dataset_type $dataset_type