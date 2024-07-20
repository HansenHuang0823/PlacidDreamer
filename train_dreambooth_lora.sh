export MODEL_NAME="../../stable-diffusion-2-1-base"
export INSTANCE_DIR="./multiview_images/cactus"
export OUTPUT_DIR="./lora_checkpoints/cactus"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="A cactus with pink flowers" \
  --class_prompt="A cactus" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --pre_compute_text_embeddings
