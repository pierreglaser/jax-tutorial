## Installation Instructions


### Requirements:

- A UNIX-Compliant distribution
- A `conda`-based package manager
- (Optional) for GPU support: CUDA driver libraries `>= 11.6`.

### Jax Installation (CPU)

To use a CPU-only powered jax, create a `conda` virtual environment containing `python` and `jax`:
```bash
conda create -n jax-tutorial python=3.9 && conda activate jax-tutorial
conda install -c conda-forge numpy scipy jax flax numpyro
```

### Jax Installation (GPU)

In all cases, you will need to install a GPU-able version of jax.

```bash
# Installs the wheel compatible with CUDA 11 and cuDNN 8.2 or newer.
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

A fully-functionning version of jax (i.e which includes working working (sparse) linear algebra and deep network primitives) on GPU requires `cudatoolkit` libraries, `cudnn`, as well as  `nvcc` (a CUDA compiler).
In most cases, these libraries should already be present in your system. Alas for research staff working on compute clusters with only user privileges, they often reside in a non-standard locations.

#### If CUDA-related utilities are available in standard locations
You should be all set. Congrats for living such a luxurious life.

#### If using properly configured modulefiles (case of the Sainsbury Wellcome Center Compute Cluster).
Some compute environments (like the SWC compute cluster) use modulefiles to integrate specific libraries and executables with your current shell session, removing the need for environment variables plumbing when the said libraries/executables are present in non-standard locations.

If you're a SWC staff researcher working on the SWC compute cluster, you can load the cuda/11.6 modulefile by executing:

```bash
module load cuda/11.6
```

and *voila*.


#### If CUDA-related utilities are available in a non-standard locations
If none of the two cases above apply, for instance in the case of user (conda) installed  CUDA-libraries, or incomplete module files, you will need to point to `jax` yourself  the place where such libraries can be found.
To do so, locate the root directory containing the cuda utilities, say, `/path/to/cuda`, and run:

```bash
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda/dir;
export LD_LIBRARY_PATH=/path/to/cuda/dir/lib64;  # YMMV: might be lib and not lib64
```


### Testing your installation

To test that your jax environment is properly setup, a convenience script is provided as part of this tutorial. From the root directory of this repository run:
```bash
python -m pip install ./jax-utils
# if on CPU:
python -m jax_utils.test_jax_installation
# if on GPU:
python -m jax_utils.test_jax_installation --gpu
```

This script will test a subset of jax features relying on different libraries and will loudly error out if some piece of software is missing.


### Installing jupyter-related utilities

To execute jupyter notebooks that will use the previously setup `jax-tutorial` environment as the execution environment, either install `jupyterlab` directly in this environment:

```bash
conda install jupyterlab
```

or install `ipykernel` and register your kernel to your external jupyterlab installation:

```bash
conda install ipykernel
python -m ipykernel install --prefix=path/to/miniforge/installation/envs/<jupyterlab-installation-env-name> --name="jax-tutorial";
conda deactivate && conda activate <jupyterlab-installation-env-name>
```

If you're using a GPU-powered jax, jupyterlab, and you're feeling fancy, install the jupyterlab extension `jupyterlab_nvdashboard`, which will dynamically display
valuable metrics such as GPU memory usage or GPU volatle utilisation:

```bash
pip install jupyterlab_nvdashboard
```

At this point, you should bee all set. To execute the notebooks `tutorial.ipynb`, simply make sure you are in the root directory of this tutuorial's repository, and execute:

```bash
jupyter lab
```