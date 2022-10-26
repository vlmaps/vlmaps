# VLMaps: Visual Language Maps for Robot Navigation

## DEMO
You can try our code with the CoLab notebook [Here](https://colab.research.google.com/drive/1xsH9Gr_O36sBZaoPNq1SmqgOOF12spV0?usp=sharing).
If you want to run our code locally, you could follow the instructions below.

## Dependencies Installation

Our code is tested in the Ubuntu 20.04 system. We recommend using Conda for python package management. Please install conda with the following commands if you don't have it on your machine:

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p <PATH WHERE CONDA IS INSTALLED>
rm -rf Miniconda3-latest-Linux-x86_64.sh
```

You can install the dependencies with the `setup.bash` file or run the following commands in a terminal manually.

### install with `setup.bash` file
```bash
chmod +x setup.bash
./setup.bash
```

### install manually

```bash
# installation for lseg
conda create -n vlmaps python==3.8 -y
conda activate vlmaps
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
conda install habitat-sim -c conda-forge -c aihabitat
cd ~
git clone https://github.com/facebookresearch/habitat-lab.git
pip install gym==0.22.0
cd habitat-lab
git checkout bfba72f47800819d858a6859b14cfa26122c2762
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
pip install pyvisgraph
```

