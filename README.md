# JAX Barnes-Hut

This repository contains an implementation of a number of Barnes Hut related algorithms designed for GPU acceleration, backpropogation through the tree, and integration into JAX-based frameworks. 

This work is derived from a reference implementation for the algorithm described in the
SIGGRAPH 2025 paper, "Stochastic Barnes-Hut Approximation for Fast Summation on
the GPU". The project page can be found
[here](https://www.dgp.toronto.edu/projects/stochastic-barnes-hut/).

For dynamical applications, the stochastic implementations are still under developments.

## Prerequisites
Make sure you have Python installed, and set up an environment (using, e.g.,
Conda) with the following libraries:
- `numpy`
- `gpytoolbox`
- `libigl`
- `polyscope`
- `robust_laplacian`
- `jax`

Certain related dynamical/cosmological examples require some combination of 
- `diffrax`
- `jaxpm`

## Build Instructions
```
mkdir build
cd build
cmake ..
make -j2
```

This will produce a `.so` library on Unix-based systems (or two on systems with
an NVIDIA GPU), so copy those files to the `python` directory. After that, try
to run `example.py`. Also pass in the `-h` flag to see different options, like
running on the GPU and using different kernels.

On Linux, you may need to run the following command to get your Conda
environment to work correctly (from [this SO
post](https://stackoverflow.com/a/79286982)):
```
conda install -c conda-forge libstdcxx-ng
```

## Example Use

We provide a test script to drive and benchmark the various implementation in ./python/unified_force_benchmark.py 

Exact performance characteristics of the various implementations will depend heavily on the various factors used; tree depth, opening angle, etc. The octree8 is particularly finicky; more depth usually increases error but dramatically speeds things up.


N = 10,000

method                   | status | rel_l2    | outlier_frac | compute_ms | compile_ms

cpu_numpy_direct         | ok     | 0.000e+00 | 0.000        | 5324.467   | NA        
jax_gpu_direct_no_custom | ok     | 1.524e-06 | 0.000        | 2.947      | 271.787   
ffi_cuda_direct          | ok     | 1.279e-07 | 0.000        | 4.863      | NA        
classic_bh_l3            | ok     | 2.705e-06 | 0.000        | 8.508      | NA        
classic_bh_octree8       | ok     | 1.538e-02 | 0.062        | 22.686     | NA        
stochastic_bh_l3         | ok     | 2.705e-06 | 0.000        | 7.877      | NA        
stochastic_bh_octree8    | ok     | 2.374e-06 | 0.000        | 9.485      | NA        

N=1,000,000 (impossible for pure JAX and CPU on my system)

method                | status | rel_l2    | outlier_frac | compute_ms | compile_ms

ffi_cuda_direct       | ok     | 0.000e+00 | 0.000        | 12766.763  | NA        
classic_bh_l3         | ok     | 2.536e-05 | 0.000        | 13249.489  | NA        
classic_bh_octree8    | ok     | 1.832e-02 | 0.083        | 4888.022   | NA        
stochastic_bh_l3      | ok     | 2.536e-05 | 0.000        | 15256.959  | NA        
stochastic_bh_octree8 | ok     | 4.588e+00 | 0.752        | 1420.857   | NA  

(Note the current stochastic_bh_octree8 blows up)

## Brief Guide to Numerics...

## Future Work
- [ ] Further optimize Stochastic Kernel for dynamical applications
- [ ] Debug or properly control stochastic_bh_octree8 implementation at larger tree depths.
- [ ] Explore possible stochastic-friendly integration schemes for particle motion.
