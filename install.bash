#!/bin/bash
pip install -r requirements.txt

cd ~
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization/
python -m pip install -e .
