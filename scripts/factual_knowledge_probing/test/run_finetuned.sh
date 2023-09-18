out_dir=$1
dataset_type=$2

nohup python -m src.factual_knowledge_probing.run_factual_knowledge_probing \
    --model_name_or_path $out_dir \
    --do_train False \
    --do_eval True \
    --validation_file "./data/LAMA_TREx/"$dataset_type".json" \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --fp16 True \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --block_size 128 \
    --save_strategy no \
    --seed 0 \
    --report_to tensorboard \
    --output_dir $out_dir \
    --truncated_prompt False \
    > $out_dir/eval_LAMA_TREx_$dataset_type.log &