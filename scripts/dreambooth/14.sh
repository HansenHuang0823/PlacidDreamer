export CUDA_VISIBLE_DEVICES="6"
export INSTANCE_DIR="./T3Bench_multiview/14"
export OUTPUT_DIR="./lora_checkpoints/14"
export INSTANCE_PROMPT="A plush teddy bear with a satin bow"
export CLASS_PROMPT="A teddy bear"
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="../../stable-diffusion-2-1-base"  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="$INSTANCE_PROMPT" \
  --class_prompt="$CLASS_PROMPT" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --pre_compute_text_embeddings