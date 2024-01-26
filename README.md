# Working through Fast.ai example from lecture 2

For my local system, I installed requirements using mamba after installing
the pip version of pytorch for GPU:

```sh
pip uninstall torch
pip cache purge
pip install torch -f https://download.pytorch.org/whl/torch_stable.html
```

Install miniforge and Mamba: [Mamba installation](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)

```sh
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Then create a mamba environment and install pytorch using [pytorch getting started](https://pytorch.org/get-started/locally/)

```sh
mamba create -n pytorch python=3.10
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
mamba install jupyter
```

Test if CUDA is working

```bash
mamba activate pytorch
python -c 'import torch; print(torch.cuda.is_available())'
True
```
