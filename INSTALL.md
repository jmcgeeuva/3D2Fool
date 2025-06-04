# Install

1. Start a rivanna GPU instance
1. Create your conda environment
``` bash
# https://github.com/Gandolfczjh/3D2Fool/issues/17
conda create -n cars python=3.8.17 nvidiacub fvcore iopath  pytorch=1.13.0 torchvision pytorch-cuda=11.6 jupyter -c pytorch -c nvidia -c bottler -c fvcore -c iopath -c conda-forge
source activate cars
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
nvcc --version
```
1. Add cub
```bash
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
export CUB_HOME=$PWD/cub-1.10.0
```
1. Install pytorch3d
```bash
# https://github.com/facebookresearch/pytorch3d/blob/v0.7.4/INSTALL.md
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
conda install cudatoolkit-dev -c conda-forge
python setup.py install
python -c "import pytorch3d"
cd ..
```
4. Install python packages for 3D2Fool
```bash
python -m pip install opencv-python matplotlib tqdm
```
5. Clone 3D2Fool and run the script
```bash
git clone https://github.com/jmcgeeuva/3D2Fool.git
python attack_base.py --train_dir ./ --device 'cuda'
```