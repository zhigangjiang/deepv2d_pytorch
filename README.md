# DeepV2D (PyTorch)

This is PyTorch implementation of the paper "[DeepV2D: Video to Depth with Differentiable Structure from Motion](https://arxiv.org/abs/1812.04605)"(ICLR 2020). 

Original tensorflow project is at [here](https://github.com/princeton-vl/DeepV2D).

# Installation
Develop environment:
```angular2html
OS: Ubuntu 20.04.4
CUDA: 11.4
```

Install dependencies:

```shell
pip install -r requirements.txt
```

Install [LieTorch](https://github.com/princeton-vl/lietorch) for SE3 operations to pose operation:
```shell
cd lietorch
python setup.py install
./run_tests.sh # for test
```