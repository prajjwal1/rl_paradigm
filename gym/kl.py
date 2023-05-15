import pickle

load_file = "/checkpoints/prajj/gym/humanoid-medium-v2.pkl"

with open(load_file, "rb") as fp:
    data_1 = pickle.load(fp)

load_file = "/checkpoints/prajj/gym/humanoid-expert-v2.pkl"

with open(load_file, "rb") as fp:
    data_2 = pickle.load(fp)

data_1.extend(data_2)

load_file = "/checkpoints/prajj/gym/humanoid-medium-expert-v2.pkl"
with open(load_file, "wb") as fp:
    pickle.dump(data_1, fp, protocol=pickle.HIGHEST_PROTOCOL)
