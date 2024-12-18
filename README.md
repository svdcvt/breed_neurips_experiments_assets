## How to install APEBench _for GPU_

Before installing, make sure `LD_LIBRARY_PATH` is not set.

There is a `jax[cuda12]`, which should install its own cuda toolkit and cuDNN version from python wheels. So, you can remove `cuda cudnn` from `conda install`.
It didn't work for me. So, I did it manually and let jax use that instead,

```bash
conda create -n apebench python=3.10 cuda cudnn
conda activate apebench
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade "jax[cuda12_local]"
python3 -m pip install apebench
```

Now, this might not be able to detect the GPU, which is still a mystery.
But, after restarting my computer, I was able to get this.
```bash
(apebench) abhishek@local:~/Projects/apebench_test$ python3
Python 3.10.16 (main, Dec 11 2024, 16:24:50) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import jax
>>> jax.devices()
[CudaDevice(id=0)]
```

Set `XLA_FLAGS` for cuda library search path to find `nvvm/libdevices.so`,
```bash
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX"
```

Finally, try a toy training with apebench,
```bash
python3 hello.py
```

