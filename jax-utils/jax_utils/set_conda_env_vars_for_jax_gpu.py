import os
import sys
from pathlib import Path


def set_environment_for_jax():
    try:
        __import__("jax")
    except ModuleNotFoundError:
        return

    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        return

    # XXX: this script has been written with condainstallation in mind --
    # I should maybe check whether this python belongs to a conda env.

    conda_env_bin_dir = Path(sys.executable).parent
    conda_env_dir = conda_env_bin_dir.parent
    conda_env_lib_dir = conda_env_dir / "lib"

    # TODO(piereglaser): expose cuda toolkit to jax
    os.environ["PATH"] = f"{os.environ['PATH']}:{conda_env_bin_dir}"

    if "XLA_FLAGS" not in os.environ:
        print("setting XLA_FLAGS")
        os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={conda_env_dir}"

    if "LD_LIBRARY_PATH" not in os.environ:
        print("setting LD_LIBRARY_PATH")
        os.environ["LD_LIBRARY_PATH"] = f"{conda_env_lib_dir}"

    # Don't prealloacte 90% of GPU memory as it recently led to memory leaks
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".5"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
