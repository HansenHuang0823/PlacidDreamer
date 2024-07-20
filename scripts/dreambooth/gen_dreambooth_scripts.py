import os
with open("train_dreambooth_lora.sh", "r") as f:
    data = f.readlines()

with open("./prompt_single.txt", "r") as f:
    instance_prompt_list = f.readlines()

with open("./prompt_single_class.txt", "r") as f:
    class_prompt_list = f.readlines()

# export INSTANCE_DIR="./multiview_output"
# export OUTPUT_DIR="./lora_checkpoints/400"
# export INSTANCE_PROMPT="A clever chimpanzee dressed like Henry VIII king of England."
# export CLASS_PROMPT="A chimpanzee dressed like Henry VIII king of England."

for index in range(100):
    case_data = []
    case_data.append('export CUDA_VISIBLE_DEVICES="' + str(index % 8) + '"\n')
    case_data.append('export INSTANCE_DIR="' + os.path.join("./T3Bench_multiview", str(index)) + '"\n')
    case_data.append('export OUTPUT_DIR="' + os.path.join("./lora_checkpoints", str(index)) + '"\n')
    case_data.append('export INSTANCE_PROMPT="' + str(instance_prompt_list[index]).strip() + '"\n')
    case_data.append('export CLASS_PROMPT="' + str(class_prompt_list[index]).strip() + '"\n')
    case_data = case_data + data
    with open(str(index) + ".sh", "w") as f:
        f.writelines(case_data)