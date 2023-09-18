model_name_or_path=results/gpt_neo_1_3B
out_dir=gpt_neo_1_3B_TREx

nohup deepspeed src/factual_knowledge_probing/run_factual_knowledge_probing.py \
    --deepspeed src/utils/ds_config_zero3.json \
    --model_name_or_path $model_name_or_path \
    --do_train True \
    --do_eval True \
    --train_file "./data/LAMA_TREx/train.json" \
    --validation_file "./data/LAMA_TREx/test.json" \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
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