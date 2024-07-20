for gpu_id in range(8):
    gpu_work = []
    for j in range(100 // 8 + 1):
        index = 8 * j + gpu_id
        if index >= 100:
            continue
        gpu_work.append('python train.py --opt "./configs/{}.yaml" --name "{}" --lora_path "./lora_checkpoints/{}"\n'.format(index, index, index))
    with open("{}".format(gpu_id) + ".sh", "w") as f:
        f.writelines(gpu_work)