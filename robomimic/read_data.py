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
    cnt = 0
    for partial_f in fnames:
        temp_res = 0
        try:
            f = open(args.json_path + partial_f + "/results.json")
            json_dict = json.load(f)
            #  print(json_dict)
            for level1, val in json_dict.items():
                for task_name, data in val.items():
                    temp_res = max(temp_res, data["Success_Rate"])
            cnt += 1
            res.append(temp_res)
        except FileNotFoundError as e:
            print(f"Following file not present: {partial_f}")
    return cnt, res


cnt, res = read_data(args)

print(f"Mean: {np.mean(res)}, Std: {np.std(res)}")
print(f"Results coming from {cnt//3} observations")
#  print("Total number of elements: ", len(res))
#  print('Average Normalized Score: ', np.average(res))
#  print('Std Dev: ', np.std(res))
#  print("Trainable params: ", trainable_params)
#  print("Training time: ", training_time)
#  print("Eval time: ", eval_time)
