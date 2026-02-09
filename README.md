## Multi-Modal Whole-Body Control Framework 

### env setup

please follow the instructions in [env_setup.md](docs/env_setup.md) to setup the environment.

### Snapshot



### Quick Start

#### 1. Download datasets

We provide the preprocessed dataset for training and evaluation, which can be downloaded from [here](https://www.modelscope.cn/datasets/seulzx/gae_mimic_dataset). After downloading, please unzip the dataset and place it in the `datasets` directory.

The file structure should look like this:

```
whole_body_control/
├── datasets/
│   ├── extended_datasets/
│   │   ├── lafan1_dataset/
│   │   ├── 100style_dataset/
│   |   |      ├── g1/
│   |   |      ├── info.yaml
```

#### 2. Train

```
python scripts/rsl_rl/train.py --headless --task MultiTracking-Flat-G1-v0
```
