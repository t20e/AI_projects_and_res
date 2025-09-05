# Creating Conda Env

### Rules

- Always try to install packages with conda first (conda install -c conda-forge <package_name>).

- Only use pip inside a Conda environment to install packages that are not available on any Conda channel.

### Creating AI Environment

```shell
# create the env
conda create -n env_name python=3.12 -y;

# Activate it
conda activate env_name;

# Install a full suite of packages commonly used in pytorch ML.
conda install -c conda-forge pytorch torchvision torchaudio;

```

**Only install below if needed**

```shell
# Install jupyter lab
conda install -c conda-forge jupyterlab;

# Install pandas & matplotlib packages
conda install -c conda-forge pandas matplotlib;

# Install scikit-learn 
conda install -c conda-forge scikit-learn;

# HDF5 lets you store huge amounts of numerical data, and easily manipulate that data from NumPy.
conda install -c conda-forge h5py;

```
