#!/bin/bash
pip install -r requirements.txt

conda install habitat-sim=0.2.2 -c conda-forge -c aihabitat -y

cd ~
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization/
python -m pip install -e .
