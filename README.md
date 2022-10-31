# VLMaps
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xsH9Gr_O36sBZaoPNq1SmqgOOF12spV0?usp=sharing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


[<b>Visual Language Maps for Robot Navigation</b>](https://arxiv.org/pdf/2210.05714.pdf)

[Chenguang Huang](http://www2.informatik.uni-freiburg.de/~huang/), [Oier Mees](https://www.oiermees.com/), [Andy Zeng](https://andyzeng.github.io/), [Wolfram Burgard](http://www2.informatik.uni-freiburg.de/~burgard)

We present **VLMAPs** (**V**isual **L**anguage **Maps**),  a spatial map representation in which pretrained visuallanguage model features are fused into a 3D reconstruction of the physical
world. Spatially anchoring visual language features enables *natural language indexing in the map*, which can be used to, e.g., localize landmarks
or spatial references with respect to landmarks – enabling zero-shot spatial
goal navigation without additional data collection or model finetuning.

![](media/banner.png)

# Quick Start
Try VLMaps creation and landmark indexing in [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xsH9Gr_O36sBZaoPNq1SmqgOOF12spV0?usp=sharing).

To begin on your own machine, clone this repository locally
```bash
git clone https://github.com/vlmaps/vlmaps.git

```
Install requirements:
```bash
$ conda create -n vlmaps python=3.8  # or use virtualenv
$ conda activate vlmaps
$ bash install.bash
```
Start the jupyter notebook
```bash
$ cd vlmaps
$ jupyter notebook demo.ipynb
```


# Benchmark



## Citation

If you find the dataset or code useful, please cite:

```bibtex
@article{huang22vlmaps,
  title     = {Visual Language Maps for Robot Navigation},
  author    = {Chenguang Huang and Oier Mees and Andy Zeng and Wolfram Burgard},
  journal   = {arXiv preprint arXiv:2210.05714},
  year      = {2022},
}
```

## License

MIT License
