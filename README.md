# PlacidDreamer
The official implementation of ACM Multimedia 2024 paper "PlacidDreamer: Advancing Harmony in Text-to-3D Generation".

### Pretrained Weights
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