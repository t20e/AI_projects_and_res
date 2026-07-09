# Creating A Conda Environment

## Rules

- Always try to install packages with conda first (conda install -c conda-forge <package_name>), except PyTorch and its necessary packages use Pip instead.

- Only use pip inside a Conda environment to install packages that are not available on any Conda channel.

## Creating AI Environment

```shell
# create the env
conda create -n env_name python=3.14 -y;

# Activate it
conda activate env_name;
```

## Install Below If Needed

```shell
# Install a full suite of packages commonly used in pytorch ML.
pip install torch torchvision torchaudio numpy matplotlib pandas

# ipykernel is needed for jupyter notebooks
conda install conda-forge::ipykernel
```

```shell
# Install jupyter lab
conda install -c conda-forge jupyterlab;

# Install scikit-learn 
conda install -c conda-forge scikit-learn;

# HDF5 lets you store huge amounts of numerical data, and easily manipulate that data from NumPy.
conda install -c conda-forge h5py;

```
