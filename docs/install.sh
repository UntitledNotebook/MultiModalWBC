conda create -n m3imic python=3.10 -y
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
cd source/whole_body_control
pip install -e .


