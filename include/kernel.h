#pragma once

#define _USE_MATH_DEFINES
#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "leaf_data.h"
#include "utils.h"

// Per-translation-unit periodic settings used by FFI kernels.
// Defaults to non-periodic.
#ifdef __CUDACC__
#define SBH_DEVICE_CONST __device__ __constant__
#else
#define SBH_DEVICE_CONST
#endif
static SBH_DEVICE_CONST int g_periodic_min_image = 0;
static SBH_DEVICE_CONST float g_periodic_box_length = 1.0f;

template<int N>
BOTH VectorNd<N> minImageDisplacement(VectorNd<N> dvec) {
#ifdef __CUDA_ARCH__
  if (g_periodic_min_image != 0 && g_periodic_box_length > FP(0.0)) {
    FLOAT inv_l = FP(1.0) / static_cast<FLOAT>(g_periodic_box_length);
    for (int i = 0; i < N; i++) {
      dvec(i) = dvec(i) - static_cast<FLOAT>(g_periodic_box_length) *
                            floor(dvec(i) * inv_l + FP(0.5));
    }
  }
#endif
  return dvec;
}

// TODO: eventually change this to take in a third "extra kernel data"
// parameter, or perhaps just a generic block of memory that each kernel can
// interpret/ignore as needed.
template <typename T, int N>
using KernelFunc = T (*)(const LeafData<N>&, VectorNd<N>);

// A list of supported kernel functions for communication with the plugin.
enum KernelFuncID {
  GRAVITY,
  WINDING_NUMBER,
  SMOOTH_DIST
};

// Kernel functions
template<int N>
BOTH FLOAT gravityPotential(const LeafData<N>& data, VectorNd<N> q) {
  VectorNd<N> dvec = minImageDisplacement<N>(data.position - q);
  FLOAT dist = dvec.norm();
  return -data.mass / (dist + FP(1e-1));
}

template<int N>
BOTH FLOAT signedSolidAngle(const LeafData<N>& data, VectorNd<N> q);

template<>
BOTH FLOAT signedSolidAngle<2>(const LeafData<2>& data, VectorNd<2> q) {
  VectorNd<2> dvec = minImageDisplacement<2>(data.position - q);
  FLOAT dist_sq = dvec.squaredNorm();
  FLOAT q_dot_n = data.normal.dot(dvec);
  return data.mass * q_dot_n / (2 * (FLOAT)PI * (dist_sq + FP(1e-5)));
}

template<>
BOTH FLOAT signedSolidAngle<3>(const LeafData<3>& data, VectorNd<3> q) {
  VectorNd<3> dvec = minImageDisplacement<3>(data.position - q);
  FLOAT dist = dvec.norm();
  FLOAT q_dot_n = data.normal.dot(dvec);
  return data.mass * q_dot_n /
         (4 * (FLOAT)PI * (dist * dist * dist + FP(1e-5)));
}

template <int N>
BOTH FLOAT smoothDistExp(const LeafData<N>& data, VectorNd<N> q) {
  VectorNd<N> dvec = minImageDisplacement<N>(data.position - q);
  FLOAT dist = dvec.norm();
  return exp(-data.alpha * dist);
}

template<int N>
BOTH VectorNd<N> gravityGrad(const LeafData<N>& data, VectorNd<N> q) {
  VectorNd<N> dvec = minImageDisplacement<N>(q - data.position);
  FLOAT dist = dvec.norm();
  return -data.mass / std::pow(dist + FP(1e-1), 3) * dvec;
}

// Fixed frequency, but can bundle in leaf data later
template<int N>
BOTH FLOAT helmholtzKernelX(const LeafData<N>& data, VectorNd<N> q) {
  FLOAT freq = 2;
  VectorNd<N> dvec = minImageDisplacement<N>(data.position - q);
  FLOAT dist = dvec.norm();
  return data.mass * std::cos(freq * 2 * (FLOAT)PI * dist) /
         (dist + FP(1e-1));
}

template<int N>
BOTH FLOAT helmholtzKernelY(const LeafData<N>& data, VectorNd<N> q) {
  FLOAT freq = 1;
  VectorNd<N> dvec = minImageDisplacement<N>(data.position - q);
  FLOAT dist = dvec.norm();
  return data.mass * std::sin(freq * 2 * (FLOAT)PI * dist) /
         (dist + FP(1e-1));
}
