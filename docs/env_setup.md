# Conda / Mamba Environment Setup for Isaac Sim

---

## 1. conda / mamba installation

### choose conda, miniconda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### choose mamba, a faster conda alternative
```
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
---

## 2. conda or mamba environment setup

### check gpu driver support


```
conda create -n env_mimic python=3.10 
# or mamba create -n env_mimic python=3.10
conda activate env_mimic 
```

```
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

---

## 3. isaac sim installation

### pip install isaacsim
```
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
```
### eval isaacsim, a100 not run this cmd

```
isaacsim isaacsim.exp.full.kit
```

---

## 4. isaac lab installation and eval

### install isaaclab, and switch v2.1.1

```
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 90b79bb2d44feb8d833f260f2bf37da3487180ba
./isaaclab.sh -i
```

### eval isaaclab
```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless
cd ..
```


## 5. install rsl_rl 

```
cd path/to/IsaacLab
./isaaclab.sh -p -m pip install -e path/to/rsl_rl
```

## 6. download assets

``` 
git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
```

## 7. install whole_body_control

```
cd source/whole_body_control
pip install -e .
```

## 8. datasets

