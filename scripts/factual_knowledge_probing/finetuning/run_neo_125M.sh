model_name_or_path=results/gpt_neo_125M
out_dir=gpt_neo_125M_TREx

nohup python -m src.factual_knowledge_probing.run_factual_knowledge_probing \
    --model_name_or_path $model_name_or_path \
    --do_train True \
    --do_eval True \
    --train_file "./data/LAMA_TREx/train.json" \
    --validation_file "./data/LAMA_TREx/test.json" \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing True \
    --fp16 True \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --block_size 128 \
    --save_strategy no \
    --seed 0 \
    --report_to tensorboard \
    --output_dir results/$out_dir \
    --truncated_prompt False \
    > results/log_finetuning.$out_dir &