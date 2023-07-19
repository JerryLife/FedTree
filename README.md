---
title: Documentation of DeltaBoost
numbersections: true
---

[//]: # (Table of contents)
# Contents
- [Getting Started](#getting-started)
  - [Prepare Data](#prepare-data)
    - [Install Python Environment](#install-python-environment)
    - [Download and Preprocess Datasets](#download-and-preprocess-datasets)
  - [Environment](#environment)
    - [Install NTL](#install-ntl)
    - [Install GMP](#install-gmp)
    - [Install Boost](#install-boost)
  - [Build DeltaBoost](#build-deltaboost)
- [Usage of DeltaBoost](#usage-of-deltaboost)
  - [Basic Usage](#basic-usage)
  - [Parameter Guide](#parameter-guide)
- 

[//]: # (Contents)

# Getting Started

## Prepare Data

### Install Python Environment
DeltaBoost requires `Python >= 3.9`. The required packages have been included in `python-utils/requirements.txt`. Install necessary modules by
```shell
pip install -r requirements.txt
```
### Download and Preprocess Datasets

Download datasets and remove instances from samples.
```shell
bash download_datasets.sh
```
This script will download 5 datasets from LIBSVM wesbite. After downloading and unzipping, some instances will be removed from these datasets. The removing ratio is `0.1%` and `1%` by default. The time of removal may take several minutes. If more ratios is needed, you can change the `-r` option of `remove_sample.py`.

After the preparation, there should exist a `data/` directory with the following structure.

```text
data
├── cadata
├── cadata.test
├── cadata.train
├── cadata.train.delete_1e-02
├── cadata.train.delete_1e-03
├── cadata.train.remain_1e-02
├── cadata.train.remain_1e-03
├── codrna.test
├── codrna.train
├── codrna.train.delete_1e-02
├── codrna.train.delete_1e-03
├── codrna.train.remain_1e-02
├── codrna.train.remain_1e-03
├── covtype
├── covtype.test
├── covtype.train
├── covtype.train.delete_1e-02
├── covtype.train.delete_1e-03
├── covtype.train.remain_1e-02
├── covtype.train.remain_1e-03
├── gisette.test
├── gisette.train
├── gisette.train.delete_1e-02
├── gisette.train.delete_1e-03
├── gisette.train.remain_1e-02
├── gisette.train.remain_1e-03
├── msd.test
├── msd.train
├── msd.train.delete_1e-02
├── msd.train.delete_1e-03
├── msd.train.remain_1e-02
└── msd.train.remain_1e-03
```

## Environment
The required packages for DeltaBoost includes 
* CMake 3.15 or above
* GMP
* NTL
* Boost

### Install NTL
The NTL can be installed from source by 
```shell
wget https://libntl.org/ntl-11.5.1.tar.gz
tar -xvf ntl-11.5.1.tar.gz
cd ntl-11.5.1/src
./configure SHARED=on
make -j
sudo make install
```
If `NTL` is not installed under default folder, you need to specify the category of NTL during compilation by
```shell
cmake .. -DNTL_PATH="PATH_TO_NTL"
```

### Install GMP
The GMP can be directly installed by `apt` on Debian-based Linux, e.g. Ubuntu.
```shell
sudo apt-get install libgmp3-dev
```

### Install Boost
DeltaBoost requires `boost >= 1.75.0`. Since it may not be available on official `apt` repositories, you may need to install manually.

Download and unzip `boost 1.75.0`.
```shell
wget https://boostorg.jfrog.io/artifactory/main/release/1.75.0/source/boost_1_75_0.tar.bz2
tar -xvf boost_1_75_0.tar.bz2
```
Install dependencies for building boost.
```shell
sudo apt-get install build-essential g++ python-dev autotools-dev libicu-dev libbz2-dev libboost-all-dev
```
Start building.
```shell
./bootstrap.sh --prefix=/usr/
./b2
sudo ./b2 install
```

## Build DeltaBoost
Build DeltaBoost by
```shell
mkdir build && cd build
cmake ..
make -j
```
An executable named `build/bin/FedTree-train` should be created. For convenience, you may create a symlink for this binary.
```shell
cd ..   # under root dir of DeltaBoost
ln -s build/bin/FedTree-train main
```
# Usage of DeltaBoost
For simplicity, the usage guide assumes that the binary `main` has been created.

## Basic Usage
DeltaBoost can be configured by a `.conf` file or/and the command line parameters. For example,
```shell
./main conf=conf/cadata.conf    # By .conf file
./main enable_delta=true nbr_size=10       # By parameters
./main conf=conf/cadata.conf enable_delta=true nbr_size=10  # By both methods
```
When both methods are applied, the parameters in the command line will overwrite the value in the `.conf` file.

Sure, here is a brief parameter guide in markdown format.

## Parameter Guide

- **dataset_name** (std::string)
    - Usage: The name of the dataset.
    - Default value: ""

- **save_model_name** (std::string)
    - Usage: The name to save the model as.
    - Default value: ""

- **data** (std::string)
    - Usage: Path to the training data.
    - Default value: "../dataset/test_dataset.txt"

- **test_data** (std::string)
    - Usage: Path to the test data.
    - Default value: ""

- **remain_data** (std::string)
    - Usage: Path to the remaining training data after deletion.
    - Default value: ""

- **delete_data** (std::string)
    - Usage: Path to the deleted training data.
    - Default value: ""

- **n_parties** (int)
    - Usage: The number of parties in the federated learning setting.
    - Default value: 2

- **mode** (std::string)
    - Usage: The mode of federated learning (e.g., "horizontal" or "centralized").
    - Default value: "horizontal"

- **privacy_tech** (std::string)
    - Usage: The privacy technique to use (e.g., "he" or "none").
    - Default value: "he"

- **learning_rate** (float)
    - Usage: The learning rate for the gradient boosting decision tree.
    - Default value: 1

- **max_depth** (int)
    - Usage: The maximum depth of the trees in the gradient boosting decision tree.
    - Default value: 6

- **n_trees** (int)
    - Usage: The number of trees in the gradient boosting decision tree.
    - Default value: 40

- **objective** (std::string)
    - Usage: The objective function for the gradient boosting decision tree (e.g., "reg:linear").
    - Default value: "reg:linear"

- **num_class** (int)
    - Usage: The number of classes in the data.
    - Default value: 1

- **tree_method** (std::string)
    - Usage: The method to use for tree construction (e.g., "hist").
    - Default value: "hist"

- **lambda** (float)
    - Usage: The lambda parameter for the gradient boosting decision tree.
    - Default value: 1

- **verbose** (int)
    - Usage: Controls the verbosity of the output.
    - Default value: 1

- **enable_delta** (std::string)
    - Usage: Enable or disable the delta boosting parameter ("true" or "false").
    - Default value: "false"

- **remove_ratio** (float)
    - Usage: The ratio of data to be removed in delta boosting.
    - Default value: 0.0

- **min_diff_gain** (int)
    - Usage: (Please provide the usage)
    - Default value: ""

- **max_range_gain** (int)
    - Usage: (Please provide the usage)
    - Default value: ""

- **n_used_trees** (int)
    - Usage: The number of trees to be used in delta boosting.
    - Default value: 0

- **max_bin_size** (int)
    - Usage: The maximum bin size in delta boosting.
    - Default value: 100

- **nbr_size** (int)
    - Usage: The neighbor size in delta boosting.
    - Default value: 1

- **gain_alpha** (float)
    - Usage: The alpha parameter for the gain calculation in delta boosting.
    - Default value: 0.0

- **delta_gain_eps_feature** (float)
    - Usage: The epsilon parameter for the gain calculation with respect to features in delta boosting.
    - Default value: 0.0

- **delta_gain_eps_sn** (float)
    - Usage: The epsilon parameter for the gain calculation with respect to sample numbers in delta boosting.
    - Default value: 0.0

- **hash_sampling_round** (int)
    - Usage: The number of rounds for hash sampling in delta boosting.
    - Default value: 1

- **n_quantized_bins** (int)
    - Usage: The number of quantized bins in delta boosting.
    - Default value: ""

- **seed** (int)
    - Usage: The seed for random number generation.
    - Default value: ""


## Reproduce Main Results
Before reproducing the main results, please make sure that the binary `main` has been created. All the time reported are done on two AMD EPYC 7543 32-Core Processor using 96 threads. If your machine does not have the required threads, you may
- reduce the number of seeds, for example, to `5`. However, this increases the variance of the calculated Hellinger distance.
- reduce the require threads, for example, to `taskset -c 0-11`. However, this increases the running time. If you want to use all the threads, simply remove `taskset -c 0-x` before the command.

### Removing in one tree (Table 4,5)
To test removing in a single tree with Deltaboost, simply run

```shell
bash test_remove_deltaboost_tree_1.sh 100  # try 100 seeds
```
This script finishes in **6 hours**. After the execution, two folders will appear under the project root:

- `out/remove_test/tree1` contains accuracy of each model on five datasets.
- `cache/` contains two kinds of information:
  - original model, deleted model, and retrained model in `json` format.
  - detailed per-instance prediction in `csv` format. This information is used to calculate the Hellinger distance.

To extract the information in a latex table, run

```shell
# in project root
cd python-utils
python plot_results.py -t 1
```
The scripts extracts the **accuracy** and **Hellinger distance** of DeltaBoost into Latex table. The cells of baselines to be manually filled in are left empty in this table.

Two files of summarized outputs are generated in `out/`:
- `out/accuracy_table_tree1.csv`: Results of accuracy in Table 4. An example is shown below.

```csv
,,0.0874\textpm 0.0002,,,0.0873\textpm 0.0005
,,0.0874\textpm 0.0002,,,0.0873\textpm 0.0005
,,0.0873\textpm 0.0002,,,0.0872\textpm 0.0007
,,0.2611\textpm 0.0001,,,0.2610\textpm 0.0001
,,0.2611\textpm 0.0001,,,0.2611\textpm 0.0001
,,0.2611\textpm 0.0001,,,0.2610\textpm 0.0000
,,0.0731\textpm 0.0020,,,0.0787\textpm 0.0042
,,0.0731\textpm 0.0020,,,0.0786\textpm 0.0043
,,0.0731\textpm 0.0020,,,0.0790\textpm 0.0043
-,-,0.1557\textpm 0.0034,-,-,0.1643\textpm 0.0066
-,-,0.1557\textpm 0.0034,-,-,0.1643\textpm 0.0065
-,-,0.1558\textpm 0.0034,-,-,0.1644\textpm 0.0066
-,-,0.1009\textpm 0.0003,-,-,0.1009\textpm 0.0003
-,-,0.1009\textpm 0.0003,-,-,0.1009\textpm 0.0003
-,-,0.1009\textpm 0.0003,-,-,0.1009\textpm 0.0003
```

- `out/forget_table_tree1.csv`: Results of Hellinger distance in Table 5. An example is shown below.

```csv
,,0.0002\textpm 0.0051,,,0.1046\textpm 0.2984
,,0.0000\textpm 0.0014,,,0.0070\textpm 0.0515
,,0.0162\textpm 0.1260,,,0.0300\textpm 0.1521
,,0.0000\textpm 0.0005,,,0.0069\textpm 0.0467
,,0.0007\textpm 0.0022,,,0.0070\textpm 0.0081
,,0.0000\textpm 0.0004,,,0.0051\textpm 0.0065
-,-,0.0058\textpm 0.0157,-,-,0.0087\textpm 0.0113
-,-,0.0034\textpm 0.0121,-,-,0.0033\textpm 0.0048
-,-,0.0041\textpm 0.0044,-,-,0.0126\textpm 0.0101
-,-,0.0028\textpm 0.0036,-,-,0.0093\textpm 0.0079
```

These two results might be slightly different from the results in the paper due to the randomness of the training process. However, the distance between $M_d$ and $M_r$ is very small, which is consistent as the results in the paper.

### Removing in Multiple trees (Table 7)
To test removing in 10 trees with Deltaboost, simply run

```shell
bash test_remove_deltaboost_tree_10.sh 100  # try 100 seeds
```
The script finishes in **6 hours**. After the execution, two folders will appear under the project root:
- `out/remove_test/tree10` contains accuracy of each model on five datasets.
- `cache/` contains two kinds of information:
  - original model, deleted model, and retrained model in `json` format.
  - detailed per-instance prediction in `csv` format. This information is used to calculate the Hellinger distance.
  
To extract the information in a latex table, run
```shell
# in project root
cd python-utils
python plot_results.py -t 10
```
The script extracts the **accuracy** and **Hellinger distance** of DeltaBoost into Latex table. The cells of baselines to be manually filled in are left empty in this table.

Two files of summarized outputs are generated in `out/`:
- `out/accuracy_table_tree10.csv`: Results of accuracy in Table 7(a). An example is shown below.

```csv
,,0.0616\textpm 0.0011,,,0.0617\textpm 0.0010
,,0.0617\textpm 0.0011,,,0.0618\textpm 0.0010
,,0.0617\textpm 0.0011,,,0.0617\textpm 0.0010
,,0.2265\textpm 0.0069,,,0.2265\textpm 0.0069
,,0.2264\textpm 0.0069,,,0.2265\textpm 0.0068
,,0.2264\textpm 0.0067,,,0.2255\textpm 0.0066
,,0.0509\textpm 0.0043,,,0.0490\textpm 0.0038
,,0.0509\textpm 0.0043,,,0.0490\textpm 0.0038
,,0.0508\textpm 0.0041,,,0.0497\textpm 0.0046
-,-,0.1272\textpm 0.0055,-,-,0.1396\textpm 0.0068
-,-,0.1274\textpm 0.0055,-,-,0.1400\textpm 0.0068
-,-,0.1273\textpm 0.0055,-,-,0.1399\textpm 0.0072
-,-,0.1040\textpm 0.0006,-,-,0.1040\textpm 0.0006
-,-,0.1040\textpm 0.0006,-,-,0.1040\textpm 0.0006
-,-,0.1041\textpm 0.0006,-,-,0.1040\textpm 0.0005
```

- `out/forget_table_tree10.csv`: Results of Hellinger distance in Table 7(b). An example is shown below.

```csv
,,0.0130\textpm 0.0100,,,0.0088\textpm 0.0079
,,0.0129\textpm 0.0100,,,0.0089\textpm 0.0078
,,0.0112\textpm 0.0089,,,0.0118\textpm 0.0096
,,0.0112\textpm 0.0090,,,0.0118\textpm 0.0096
,,0.0106\textpm 0.0073,,,0.0312\textpm 0.0169
,,0.0106\textpm 0.0073,,,0.0312\textpm 0.0167
-,-,0.0240\textpm 0.0169,-,-,0.0247\textpm 0.0159
-,-,0.0239\textpm 0.0160,-,-,0.0249\textpm 0.0149
-,-,0.0194\textpm 0.0106,-,-,0.0249\textpm 0.0127
-,-,0.0194\textpm 0.0106,-,-,0.0248\textpm 0.0126
```

These two results might be slightly different from the results in the paper due to the randomness of the training process. However, the distance between $M_d$ and $M_r$ is very small, which is consistent as the results in the paper.

### Efficiency (Table 6)

To test the efficiency, we need to perform a clean retrain of GBDT. To train a 10-tree GBDT, run

```shell
bash test_remove_gbdt_efficiency.sh 10
```

The script retrain GBDT on five datasets with two removal ratios for one time since the GBDT is deterministic. The script finishes in **10 minutes**. After the execution, the efficiency and speedup can be summarized by
```shell
python plot_time.py -t 10
```
The expected output should be like
```text
Thunder	& DB-Train	& DB-Remove	& Speedup (Thunder) \\
 12.410	&  8.053 \textpm 3.976	 &  0.156 \textpm 0.047	 & 79.34x \\
 12.143	&  7.717 \textpm 4.134	 &  0.160 \textpm 0.035	 & 75.82x \\
 15.668	&  52.253 \textpm 4.796	 &  1.482 \textpm 2.260	 & 10.57x \\
 16.015	&  52.333 \textpm 4.107	 &  1.874 \textpm 3.364	 & 8.55x \\
 50.213	&  66.658 \textpm 7.747	 &  0.956 \textpm 0.265	 & 52.51x \\
 47.089	&  65.322 \textpm 7.235	 &  1.123 \textpm 0.259	 & 41.95x \\
 12.434	&  6.038 \textpm 5.198	 &  0.068 \textpm 0.042	 & 183.03x \\
 12.524	&  4.704 \textpm 3.282	 &  0.053 \textpm 0.037	 & 237.99x \\
 22.209	&  53.451 \textpm 3.659	 &  3.523 \textpm 0.812	 & 6.30x \\
 24.067	&  54.221 \textpm 2.952	 &  3.422 \textpm 0.700	 & 7.03x \\
```
The time may vary due to the environment and hardwares, but the speedup is consistently significant as that in the Table 6 of the paper.

We also provide a script to running the baselines: `sklearn` and `xgboost` for efficiency comparison. Note that the performance of `xgboost` vary significantly by version. For example, some versions favors high-dimensional datasets but performs slower on large low-dimensional datasets. We adopt the default version of conda `xgboost==1.5.0` in our experiments. To run the baselines, run
```shell
taskset -c 0-95 python baseline.py  # Also limit the number of threads to 96
```
This script is expected to finish in **10 minutes**.

### Accuracy (Figure 9)
The accuracy of baselines is output by the same command as testing efficiency.
```shell
python baseline.py
```
The accuracy of DeltaBoost has also recorded in the previous logs.

