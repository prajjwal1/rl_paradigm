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
        res_dict[seed_idx] = max(json_dict.values())

    return np.mean(list(res_dict.values())), np.std(list(res_dict.values()))


res = read_data(args)

print(res)
