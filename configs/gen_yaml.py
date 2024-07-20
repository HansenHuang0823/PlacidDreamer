import yaml
import os

with open("England_crown.yaml", "r") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
with open("./prompt_single.txt", "r") as f:
    text_prompt_list = f.readlines()

for index in range(100):
    data["ModelParams"]["workspace"] = str(index)
    data["GuidanceParams"]["text"] = text_prompt_list[index].strip()
    data["GenerateCamParams"]["init_mesh"] = os.path.join("./T3Bench_multiview", str(index), "mesh.obj")
    data["GenerateCamParams"]["init_prompt"] = text_prompt_list[index].strip()
    with open(str(index) + ".yaml", "w") as f:
        yaml.dump(data, f)