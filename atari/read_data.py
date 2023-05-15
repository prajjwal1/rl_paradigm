import json
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--json_path", type=str)
args = parser.parse_args()


def read_data(args):
    fnames = os.listdir(args.json_path)
    res = []
    for partial_f in fnames:
        temp_res = 0
        f = open(args.json_path + partial_f + "/results.json")
        json_dict = json.load(f)
        for k, v in json_dict.items():
            if isinstance(v, dict):
                temp_res = max(temp_res, json_dict[k]["normalized_score"])
                res.append(temp_res)
    return (
        np.array(res),
        json_dict["training_params"],
        json_dict["training_time"],
        json_dict["eval_time"],
    )


res, trainable_params, training_time, eval_time = read_data(args)

print("Total number of elements: ", len(res))
print("Average Normalized Score: ", np.average(res))
print("Std Dev: ", np.std(res))
print("Trainable params: ", trainable_params)
print("Training time: ", training_time)
print("Eval time: ", eval_time)
