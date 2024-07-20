# PlacidDreamer
The official implementation of ACM Multimedia 2024 paper "PlacidDreamer: Advancing Harmony in Text-to-3D Generation".

### 2D Score Distillation Experiments
```bash
cd Balanced_Score_Distillation
sh run.sh
```

### Pretrained Weights
Our pretrained weights can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1T4MJisgfnPx8FR9Wd61ayM6YWJpqzpDK?usp=sharing).

Put them at `Latent_Plane/checkpoints/coarse_21500.pt` and `Latent_Plane/checkpoints/fine_8100.pt`.

### Inference
```bash
### Image to multi-view images + mesh reconstruction
### GPU requirements: 32GB
### cactus.png ---> multiview_images/cactus/*.png + mesh/cacuts.obj
python imagetomesh.py --elev 10 --im_path cactus.png

### Finetune text-to-image diffusion models
### multiview_images/cactus/*.png ---> lora_checkpoints/cactus
sh train_dreambooth_lora.sh

### Balanced Score Distillation
# Standard BSD with guidance ratio that classifier : smoothing = lambda_ : 1. You can control the saturation.
python train.py --opt configs/cactus.yaml --lambda_ 18.0 --name cactus --lora_path lora_checkpoints/cactus

# A variation of BSD using SDS decomposition. This is equivalent to standard BSD with an automatically increasing lambda. While you cannot control saturation anymore, it mostly achieves satisfactory saturation.
# See explanation below.
python train.py --opt configs/cactus.yaml --auto_BSD true --name cactus --lora_path lora_checkpoints/cactus
```

### Evaluation on T3Bench
We show the training configs and scripts on T3Bench at `configs` and `scripts` folders. Due to version updates, it is necessary to modify the storage path of the program output to ensure the pipeline runs smoothly.
```bash
### Evaluation of quality.
python T3Bench_Evaluation/eval_quality.py
### Evaluation of text alignment.
python T3Bench_Evaluation/eval_blip.py
# Replace the placeholder with your own GPT-4 api-key.
python T3Bench_Evaluation/eval_alignment.py
```