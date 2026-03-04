# Update Mar 4th

- add reward function in `rewards.py` based on `env_cat.py`
- implement `_get_rewards` fuction in `g1_cat_env.py`
- visualize obstacle in `g1_cat_env_cfg.py`
- create `agents/` for training configuration
- fix missing `__init__`
- fix buggy `list_envs.py`

# Ubuntu env setup

- Works for me. May not entirely correct.

- create & download necessary environment according to `install.sh`:

```bash
conda create -n env_cat_isaaclab python=3.10 -y
conda activate env_cat_isaaclab
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

cd ./third_party
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 90b79bb2d44feb8d833f260f2bf37da3487180ba
./isaaclab.sh -i
./isaaclab.sh -p -m pip install -e ../rsl_rl
cd ..
cd ..

git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
cp -r unitree_model/ ./source/whole_body_control/third_party/unitree_model
cp -r unitree_model/ ./third_party/unitree_model
cd source/whole_body_control
pip install -e .
```

- genereate obstables in original CAT repository and copy entire `data/` to this repository. The structure should be like:

```
MultiModalWBC-CAT/
├── data/
│   ├── RandObs/
│   └── TypiObs/
├── source/
│   └── ...
└── ...
```

# TODO

- The training does not converge. Need debugging.