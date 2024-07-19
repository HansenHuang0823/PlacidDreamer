python exp_2d.py --num_steps 1200 --log_steps 72 \
--seed 1222 --lr 0.06 --phi_lr 0.0001 --use_t_phi true \
--model_path stabilityai/stable-diffusion-2-1-base \
--t_schedule random --generation_mode bsd --phi_model lora \
--lora_scale 1. --lora_vprediction false \
--prompt "an astronaut riding a horse." \
--height 512 --width 512 --batch_size 1 \
--guidance_scale 7.5 --log_progress true \
--save_x0 true --save_phi_model true --work_dir final/