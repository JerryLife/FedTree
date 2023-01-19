import numpy as np
import argparse
import re
import os
import sys
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str)
parser.add_argument('-r', '--rate', type=float)
parser.add_argument('--is-retrain', action='store_true')

args = parser.parse_args()


def load_file(file_path, n_runs, is_retrain):
    removing_time = [0 for _ in range(n_runs)]
    training_time = [0 for _ in range(n_runs)]
    if is_retrain:
        file_path += '_retrain'
    for i in range(n_runs):
        fn = file_path + "_" + str(i) + ".out"
        print("Loading file: ", fn)
        with open(fn, "r") as f:
            for line in f:
                if "removing time" in line:
                    removing_time[i] = float(re.findall("\d+\.\d+", line)[0])
                if "training time" in line:
                    training_time[i] += float(re.findall("\d+\.\d+", line)[0])
                if "Init booster time" in line:
                    training_time[i] += float(re.findall("\d+\.\d+", line)[0])
    print(rf"training time: {np.mean(training_time): .3f} \textpm {np.std(training_time):.3f}")
    if not is_retrain:
        print(rf"removing time: {np.mean(removing_time): .3f} \textpm {np.std(removing_time):.3f}")


if args.rate == 0.01:
    load_file("out/efficiency/tree10/"+args.dataset + "_deltaboost_1e-02", 10, args.is_retrain)
elif args.rate == 0.001:
    load_file("out/efficiency/tree10/"+args.dataset + "_deltaboost_1e-03", 10, args.is_retrain)

