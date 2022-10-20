import os
import subprocess
import argparse
from typing import Literal


def show_environment_variables():
    # Make sure jax knows where to look for cuda runtime libraries
    print(f"XLA_FLAGS={os.environ.get('XLA_FLAGS')}")
    print(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH')}")
    print(f"PATH={os.environ.get('PATH')}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"XLA_PYTHON_CLIENT_PREALLOCATE={os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')}")
    print(f"XLA_PYTHON_CLIENT_ALLOCATOR={os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR')}")


def _test_jax_installation_unsafe(device: Literal['cpu', 'gpu']):
    try:
        import jax
        import jax.numpy as jnp
        from jax import random
    except ModuleNotFoundError:
        raise ValueError("jax is not installed")

    assert device in ['cpu', 'gpu'], f"device must be 'cpu' or 'gpu', got {device}"

    if device == 'gpu':
        from jaxlib.xla_extension import GpuDevice
        device_cls = GpuDevice
    else:
        from jaxlib.xla_extension import Device
        device_cls = Device

    if device == "gpu":
        # Check access to cuda compiler
        print("test access to a cuda compiler...", end="")
        try:
            subprocess.check_output(["which", "ptxas"])
            # os.system("ptxas --version")
        except subprocess.CalledProcessError as e:
            raise ValueError("No cuda compiler found in $PATH") from e
        print(" OK.")

    # tell which device jax uses

    print(f"checking if jax can detect a {device} device...", end="")
    assert any(isinstance(d, device_cls) for d in jax.local_devices())
    print(" OK.")

    # create a simple jax array
    print(f"testing array creation on {device}...", end="")
    key = random.PRNGKey(0)
    x = random.normal(key, (10,))
    print(" OK.")

    # Use specialized cuda lib such as linear algebra solvers
    print(
        "testing use of specialized cuda libraries such as linear algebra solvers...",
        end="",
    )
    A = jnp.array([[0, 1], [1, 1], [1, 1], [2, 1]])
    _, _ = jnp.linalg.qr(A)

    A = jnp.eye(10)
    _, _ = jnp.linalg.eigh(A)

    print(" OK.")

    # Use cudnn primitives such as convolutions
    # (cudnn has to be installed separately)
    print("testing use of cudnn primitives...", end="")
    key = random.PRNGKey(0)
    x = jnp.linspace(0, 10, 500)
    y = jnp.sin(x) + 0.2 * random.normal(key, shape=(500,))

    window = jnp.ones(10) / 10
    _ = jnp.convolve(y, window, mode="same")
    print(" OK.")

    print("Test done, everything seems well installed.")


def test_jax_installation(device, verbose=False):
    try:
        _test_jax_installation_unsafe(device=device)
    except Exception as e:
        print('\n')
        print('\n')
        print('##################################################################')
        print('#                                                                #')
        print('#                                                                #')
        print('#              ERROR WHILE TESTING JAX INSTALLATION              #')
        print('#                                                                #')
        print('#                                                                #')
        print('##################################################################')
        print("An error occured during the test.")
        if not verbose:
            print("Run python -m jax_utils.test_jax_installation --verbose to get more information.")
        else:
            print("Here are the environment variables:")
            show_environment_variables()
            print("Here is the error message:")
            print(e)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    test_jax_installation(device="gpu" if args.gpu else "cpu", verbose=args.verbose)
