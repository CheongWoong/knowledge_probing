target_model='gpt-3.5-turbo-0301'
dataset_type='test'

python -m src.factual_knowledge_probing.openai_api.test_chatgpt --target_model $target_model --dataset_type $dataset_type