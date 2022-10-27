#!/bin/bash

# go to the link to find suitable CUDA version for torch 1.9.1 https://pytorch.org/get-started/previous-versions/
pip install git+https://github.com/openai/CLIP.git
pip install torch-encoding
conda install ipython -y
pip install pytorch-lightning
pip install opencv-python
pip install imageio
pip install ftfy regex tqdm
pip install altair
pip install --upgrade protobuf
pip install timm
pip install tensorboardX
pip install matplotlib
pip install test-tube
pip install wandb
pip install h5py

# install habitat
pip install cmake==3.14.4
pip install tensorflow-gpu==2.9.1
conda install habitat-sim -c conda-forge -c aihabitat -y
cd ~
git clone https://github.com/facebookresearch/habitat-lab.git
pip install gym==0.22.0
cd habitat-lab
git checkout v0.2.1
python setup.py develop --all

pip install openai==0.8.0
pip install open3d
pip install grad-cam
pip install networkx
pip install mpl_point_clicker
pip install ai2thor

cd ~
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization/
python -m pip install -e .

# for real world experiment
pip install rospkg

# for shortest path computing
pip install pyvisgraph
