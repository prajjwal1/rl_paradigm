import json
import numpy as np
import os
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--json_path", type=str)
args = parser.parse_args()


def read_data(args):
    fnames = os.listdir(args.json_path)
    res_dict = defaultdict(lambda: defaultdict(float))

    for seed_idx, partial_f in enumerate(fnames):
        try:
            f = open(args.json_path + partial_f + "/results.json")
        except:
            print(f"Results not found for {seed_idx}")
            continue
        json_dict = json.load(f)
        for k, v in json_dict.items():
            epoch_dict = json_dict[k]
            for k1, v1 in epoch_dict.items():
                if k1.startswith("evaluation"):
                    res_dict[seed_idx][k1] = max(res_dict[seed_idx][k1], v1)

    mean_res_dict = defaultdict(list)

    for seed_idx, value in res_dict.items():
        for k, v in value.items():
            mean_res_dict[k].append(v)

    res_dict = defaultdict(float)
    for k, v in mean_res_dict.items():
        res_dict[k + "_mean"] = np.mean(v)
        res_dict[k + "_std"] = np.std(v)

    return res_dict


res = read_data(args)

for k, v in res.items():
    print(k, v)
