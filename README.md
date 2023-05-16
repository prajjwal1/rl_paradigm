Sequence Modeling is a Robust Contender for Offline Reinforcement Learning
(NEURIPS 2023 Submission)

Each directory contains a `script` directory within it. The script within `script` directory was used to run experiments for this study. All the scripts are highly configurable. Change the parameter according to your needs.

### Running Experiments on Atari
```bash
$ cd atari
$ sbatch scripts/train_atari.sh
```

### Running Experiments on D4RL
For DT and BC, use
```bash
$ cd gym
$ sbatch scripts/train_gym.sh
```

For CQL, use
```bash
$ cd gym
$ sbatch scripts/train_cql.sh
```

### Running Experiments on Robomimic
```bash
$ cd robomimic
$ sbatch scripts/train_default.sh
```
These scripts will create five directories named `1`, `2`, etc. depending on the number of seed provided. Each of these directory will contain a `result.json` file. Each directory is accompanied by `read_data.py` or `read_data_cql.py`. After the results are dumped, these scripts can be used to get the average mean and std dev from all the runs.

```bash
python3 read_data.py --json_file_path $FILE_TO_RESULT_DIR
```
