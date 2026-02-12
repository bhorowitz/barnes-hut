#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cub/cub.cuh>
#include <curand_kernel.h>

#include "xla/ffi/api/ffi.h"

#include "barnes_hut_gpu.h"
#include "kernel.h"
#include "leaf_data.h"
#include "tree_sampler_gpu.h"
#include "tree2d.h"
#include "tree3d.h"
#include "utils.h"

namespace sbh_ffi {

using Node2 = Tree2DNode<1, 3>;
using Leaf2 = LeafData<2>;
using Node3 = Tree3DNode<1, 3>;
using Leaf3 = LeafData<3>;

using F32R1 = xla::ffi::Buffer<xla::ffi::F32, 1>;
using F32R2 = xla::ffi::Buffer<xla::ffi::F32, 2>;
using F32R1Out = xla::ffi::ResultBuffer<xla::ffi::F32, 1>;
using F32R2Out = xla::ffi::ResultBuffer<xla::ffi::F32, 2>;
using U8R1 = xla::ffi::Buffer<xla::ffi::U8, 1>;
using U8R1Out = xla::ffi::ResultBuffer<xla::ffi::U8, 1>;

inline cudaError_t SetPeriodicBox(cudaStream_t stream, int periodic,
                                  float box_length);

static constexpr int kOctreeDimSplit = 8;
static constexpr int kOctreeNumBins =
    kOctreeDimSplit * kOctreeDimSplit * kOctreeDimSplit;
static constexpr int kOctreeMaxNodes = 1 + kOctreeNumBins;

__device__ __constant__ float g_softening_scale;
__device__ __constant__ float g_cutoff_scale;

struct OctreeNode8 {
  uint32_t start;
  uint32_t end;
  uint32_t parent;
  uint32_t first_child;
  uint32_t child[8];
  uint8_t child_mask;
  uint8_t depth;
  uint8_t is_leaf;
  uint8_t pad0;
  float cx;
  float cy;
  float cz;
  float half_size;
  float diameter;
  Leaf3 contrib;
};

static_assert(sizeof(OctreeNode8) % 8 == 0, "OctreeNode8 alignment");

__global__ void octree8_histogram_kernel(
    const float* points, const float* normals, const float* masses, int n,
    const float* min_corner, float span, int depth_bits, uint32_t* leaf_codes,
    uint32_t* leaf_counts, float* leaf_mass_sums, float* leaf_pos_sums,
    float* leaf_normal_sums, float* total_mass, float* total_pos,
    float* total_normal);
__global__ void octree8_scatter_leaf_kernel(
    const float* points, const float* normals, const float* masses, int n,
    const uint32_t* leaf_codes, const uint32_t* leaf_offsets,
    uint32_t* leaf_cursors, Leaf3* leaf_out);
__global__ void octree8_reduce_level_kernel(
    const uint32_t* child_counts, const float* child_mass_sums,
    const float* child_pos_sums, const float* child_normal_sums,
    uint32_t* parent_counts, float* parent_mass_sums, float* parent_pos_sums,
    float* parent_normal_sums, int parent_count);
__global__ void octree8_build_nodes_kernel(
    OctreeNode8* nodes, Leaf3* contrib_out, const uint32_t* leaf_offsets,
    const uint32_t* level_counts, const float* level_mass_sums,
    const float* level_pos_sums, const float* level_normal_sums,
    const uint32_t* child_counts, int level, int max_level,
    uint32_t level_offset, uint32_t parent_level_offset,
    uint32_t child_level_offset, float cx, float cy, float cz,
    float half_size);
__global__ void barnes_hut_force_octree8_kernel(
    const Leaf3* leaf_data, const OctreeNode8* nodes, const float* queries,
    int num_queries, float beta, float* out);
__global__ void barnes_hut_force_octree8_soft_kernel(
    const Leaf3* leaf_data, const OctreeNode8* nodes, const float* queries,
    int num_queries, float beta, float* out);
__global__ void barnes_hut_force_octree8_soft_vjp_kernel(
    const Leaf3* leaf_data, const OctreeNode8* nodes, const float* queries,
    const float* cotangent, int num_queries, float beta, float* out);
__global__ void stochastic_bh_force_octree8_kernel(
    const Leaf3* leaf_data, const OctreeNode8* nodes, const float* queries,
    int num_queries, int samples_per_subdomain, int seed, float* out,
    bool softened);

struct Node3Layout {
  uint64_t start_;
  uint64_t end_;
  uint64_t buffer_idx_;
  uint64_t parent_idx_;
  AABB<3> bounds_;
  FLOAT diameter_;
  Leaf3 contrib_data_;
  BitMask3D<3> bitmask_;
  uint64_t child_buffer_idxs_[kOctreeNumBins];
  int num_children_;
  int depth_;
  int max_depth_;
  bool is_leaf_;
};

static_assert(sizeof(Node3Layout) == sizeof(Node3),
              "Node3 layout mismatch with Tree3DNode<1,3>");

template <int N>
__device__ VectorNd<N> gravityGradSoftCutoff(const LeafData<N>& data,
                                             VectorNd<N> q) {
  VectorNd<N> dvec = minImageDisplacement<N>(q - data.position);
  float dist_sq = dvec.squaredNorm();
  float eps = fmaxf(g_softening_scale, 0.0f);
  float inv_r = rsqrtf(dist_sq + eps * eps + 1e-12f);
  float inv_r3 = inv_r * inv_r * inv_r;
  float attenuation = 1.0f;  // Short-range force factor.
  if (g_cutoff_scale > 0.0f) {
    // Match JaxPM's long-range split exp(-k^2 r_split^2):
    // F_short/F_newton = erfc(u) + 2u/sqrt(pi) * exp(-u^2), u=r/(2 r_split)
    float dist = sqrtf(dist_sq + 1e-12f);
    float u = dist / (2.0f * g_cutoff_scale);
    float exp_u2 = expf(-(u * u));
    attenuation = erfcf(u) + (2.0f / 1.772453850905516f) * u * exp_u2;
  }
  return -data.mass * attenuation * inv_r3 * dvec;
}

__device__ __forceinline__ uint32_t morton3d(uint32_t x, uint32_t y, uint32_t z,
                                             int bits) {
  uint32_t code = 0;
  for (int i = 0; i < bits; i++) {
    code |= ((x >> i) & 1u) << (3 * i);
    code |= ((y >> i) & 1u) << (3 * i + 1);
    code |= ((z >> i) & 1u) << (3 * i + 2);
  }
  return code;
}

__device__ __forceinline__ uint32_t quantize_axis(float v, float min_v,
                                                  float span, int bits) {
  float t = (v - min_v) / span;
  t = fminf(fmaxf(t, 0.0f), 0.999999f);
  uint32_t max_bin = (1u << bits) - 1u;
  return static_cast<uint32_t>(t * static_cast<float>(max_bin + 1));
}

__device__ __forceinline__ int quantize_axis_level3(float x, float lo,
                                                     float hi) {
  float span = hi - lo;
  if (span <= 1e-12f) {
    return 0;
  }
  float t = (x - lo) / span;
  t = fminf(fmaxf(t, 0.0f), 0.999999f);
  return static_cast<int>(t * static_cast<float>(kOctreeDimSplit));
}

__global__ void build_octree_level3_histogram_kernel(
    const float* points, const float* normals, const float* masses, int n,
    const float* min_corner, const float* max_corner, int* point_bins,
    int* bin_counts, float* bin_mass_sums, float* bin_pos_sums,
    float* bin_normal_sums, float* total_mass, float* total_pos,
    float* total_normal) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  float px = points[i * 3 + 0];
  float py = points[i * 3 + 1];
  float pz = points[i * 3 + 2];
  float nx = normals[i * 3 + 0];
  float ny = normals[i * 3 + 1];
  float nz = normals[i * 3 + 2];
  float m = masses[i];

  int qx = quantize_axis_level3(px, min_corner[0], max_corner[0]);
  int qy = quantize_axis_level3(py, min_corner[1], max_corner[1]);
  int qz = quantize_axis_level3(pz, min_corner[2], max_corner[2]);
  int bin = qx + kOctreeDimSplit * (qy + kOctreeDimSplit * qz);

  point_bins[i] = bin;
  atomicAdd(&bin_counts[bin], 1);
  atomicAdd(&bin_mass_sums[bin], m);
  atomicAdd(&bin_pos_sums[bin * 3 + 0], m * px);
  atomicAdd(&bin_pos_sums[bin * 3 + 1], m * py);
  atomicAdd(&bin_pos_sums[bin * 3 + 2], m * pz);
  atomicAdd(&bin_normal_sums[bin * 3 + 0], m * nx);
  atomicAdd(&bin_normal_sums[bin * 3 + 1], m * ny);
  atomicAdd(&bin_normal_sums[bin * 3 + 2], m * nz);
  atomicAdd(&total_mass[0], m);
  atomicAdd(&total_pos[0], m * px);
  atomicAdd(&total_pos[1], m * py);
  atomicAdd(&total_pos[2], m * pz);
  atomicAdd(&total_normal[0], m * nx);
  atomicAdd(&total_normal[1], m * ny);
  atomicAdd(&total_normal[2], m * nz);
}

__global__ void build_octree_level3_scatter_leaf_kernel(
    const float* points, const float* normals, const float* masses, int n,
    const int* point_bins, const int* bin_offsets, int* bin_cursors,
    Leaf3* leaf_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  int bin = point_bins[i];
  int dst = bin_offsets[bin] + atomicAdd(&bin_cursors[bin], 1);
  Leaf3 leaf;
  leaf.position(0) = points[i * 3 + 0];
  leaf.position(1) = points[i * 3 + 1];
  leaf.position(2) = points[i * 3 + 2];
  leaf.normal(0) = normals[i * 3 + 0];
  leaf.normal(1) = normals[i * 3 + 1];
  leaf.normal(2) = normals[i * 3 + 2];
  leaf.mass = masses[i];
  leaf.alpha = 1.0f;
  leaf_out[dst] = leaf;
}

__global__ void build_octree_level3_present_kernel(const int* bin_counts,
                                                   int* bin_present) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kOctreeNumBins) {
    return;
  }
  bin_present[i] = (bin_counts[i] > 0) ? 1 : 0;
}

__global__ void build_octree_level3_root_kernel(
    Node3Layout* nodes, Leaf3* contrib, int n, const float* min_corner,
    const float* max_corner, const float* total_mass, const float* total_pos,
    const float* total_normal) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  Node3Layout root;
  root.start_ = 0;
  root.end_ = static_cast<uint64_t>(n);
  root.buffer_idx_ = 0;
  root.parent_idx_ = 0;
  root.bounds_.min_corner(0) = min_corner[0];
  root.bounds_.min_corner(1) = min_corner[1];
  root.bounds_.min_corner(2) = min_corner[2];
  root.bounds_.max_corner(0) = max_corner[0];
  root.bounds_.max_corner(1) = max_corner[1];
  root.bounds_.max_corner(2) = max_corner[2];
  float dx = max_corner[0] - min_corner[0];
  float dy = max_corner[1] - min_corner[1];
  float dz = max_corner[2] - min_corner[2];
  root.diameter_ = sqrtf(dx * dx + dy * dy + dz * dz);
  root.contrib_data_.alpha = 1.0f;
  root.contrib_data_.mass = total_mass[0];
  if (total_mass[0] > 0.0f) {
    root.contrib_data_.position(0) = total_pos[0] / total_mass[0];
    root.contrib_data_.position(1) = total_pos[1] / total_mass[0];
    root.contrib_data_.position(2) = total_pos[2] / total_mass[0];
    root.contrib_data_.normal(0) = total_normal[0] / total_mass[0];
    root.contrib_data_.normal(1) = total_normal[1] / total_mass[0];
    root.contrib_data_.normal(2) = total_normal[2] / total_mass[0];
  } else {
    root.contrib_data_.position.setZero();
    root.contrib_data_.normal.setZero();
  }
  root.bitmask_ = BitMask3D<3>();
  for (int i = 0; i < kOctreeNumBins; i++) {
    root.child_buffer_idxs_[i] = 0;
  }
  root.num_children_ = 0;
  root.depth_ = 0;
  root.max_depth_ = 1;
  root.is_leaf_ = (n <= 1);

  nodes[0] = root;
  contrib[0] = root.contrib_data_;
}

__global__ void build_octree_level3_children_kernel(
    Node3Layout* nodes, Leaf3* contrib, const int* bin_counts,
    const int* bin_offsets, const int* child_ranks, const float* min_corner,
    const float* max_corner, const float* bin_mass_sums,
    const float* bin_pos_sums, const float* bin_normal_sums) {
  int bin = blockIdx.x * blockDim.x + threadIdx.x;
  if (bin >= kOctreeNumBins) {
    return;
  }
  int count = bin_counts[bin];
  if (count <= 0) {
    return;
  }

  Node3Layout* root = &nodes[0];
  int child_idx = 1 + child_ranks[bin];

  atomicOr(reinterpret_cast<unsigned long long*>(&root->bitmask_.mask_[bin >> 6]),
           (1ull << (bin & 63)));
  root->child_buffer_idxs_[bin] = static_cast<uint64_t>(child_idx);
  atomicAdd(&root->num_children_, 1);

  int qx = bin & (kOctreeDimSplit - 1);
  int qy = (bin >> 3) & (kOctreeDimSplit - 1);
  int qz = (bin >> 6) & (kOctreeDimSplit - 1);

  float min_x = min_corner[0];
  float min_y = min_corner[1];
  float min_z = min_corner[2];
  float dx = (max_corner[0] - min_corner[0]) / static_cast<float>(kOctreeDimSplit);
  float dy = (max_corner[1] - min_corner[1]) / static_cast<float>(kOctreeDimSplit);
  float dz = (max_corner[2] - min_corner[2]) / static_cast<float>(kOctreeDimSplit);

  float child_min_x = min_x + dx * static_cast<float>(qx);
  float child_min_y = min_y + dy * static_cast<float>(qy);
  float child_min_z = min_z + dz * static_cast<float>(qz);
  float child_max_x = child_min_x + dx;
  float child_max_y = child_min_y + dy;
  float child_max_z = child_min_z + dz;

  Node3Layout child;
  child.start_ = static_cast<uint64_t>(bin_offsets[bin]);
  child.end_ = static_cast<uint64_t>(bin_offsets[bin] + count);
  child.buffer_idx_ = static_cast<uint64_t>(child_idx);
  child.parent_idx_ = 0;
  child.bounds_.min_corner(0) = child_min_x;
  child.bounds_.min_corner(1) = child_min_y;
  child.bounds_.min_corner(2) = child_min_z;
  child.bounds_.max_corner(0) = child_max_x;
  child.bounds_.max_corner(1) = child_max_y;
  child.bounds_.max_corner(2) = child_max_z;
  child.diameter_ = sqrtf(dx * dx + dy * dy + dz * dz);
  child.contrib_data_.alpha = 1.0f;
  child.contrib_data_.mass = bin_mass_sums[bin];
  if (child.contrib_data_.mass > 0.0f) {
    child.contrib_data_.position(0) =
        bin_pos_sums[bin * 3 + 0] / child.contrib_data_.mass;
    child.contrib_data_.position(1) =
        bin_pos_sums[bin * 3 + 1] / child.contrib_data_.mass;
    child.contrib_data_.position(2) =
        bin_pos_sums[bin * 3 + 2] / child.contrib_data_.mass;
    child.contrib_data_.normal(0) =
        bin_normal_sums[bin * 3 + 0] / child.contrib_data_.mass;
    child.contrib_data_.normal(1) =
        bin_normal_sums[bin * 3 + 1] / child.contrib_data_.mass;
    child.contrib_data_.normal(2) =
        bin_normal_sums[bin * 3 + 2] / child.contrib_data_.mass;
  } else {
    child.contrib_data_.position.setZero();
    child.contrib_data_.normal.setZero();
  }
  child.bitmask_ = BitMask3D<3>();
  for (int i = 0; i < kOctreeNumBins; i++) {
    child.child_buffer_idxs_[i] = 0;
  }
  child.num_children_ = 0;
  child.depth_ = 1;
  child.max_depth_ = 1;
  child.is_leaf_ = true;

  nodes[child_idx] = child;
  contrib[child_idx] = child.contrib_data_;
}

__global__ void brute_force_gravity_kernel(const float* points,
                                           const float* masses,
                                           const float* queries,
                                           int num_points,
                                           int num_queries,
                                           int dim,
                                           float* out) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_queries) {
    return;
  }

  float acc = 0.0f;
  const float* q_ptr = queries + static_cast<int64_t>(q) * dim;

  for (int i = 0; i < num_points; ++i) {
    const float* p_ptr = points + static_cast<int64_t>(i) * dim;
    float dist_sq = 0.0f;
    for (int d = 0; d < dim; ++d) {
      float diff = p_ptr[d] - q_ptr[d];
      dist_sq += diff * diff;
    }
    float dist = sqrtf(dist_sq);
    acc += -masses[i] / (dist + 1e-1f);
  }

  out[q] = acc;
}

xla::ffi::Error BruteForceGravity(cudaStream_t stream,
                                  F32R2 points,
                                  F32R1 masses,
                                  F32R2 queries,
                                  F32R1Out out) {
  auto points_dims = points.dimensions();
  auto queries_dims = queries.dimensions();
  if (points_dims.size() != 2 || queries_dims.size() != 2) {
    return xla::ffi::Error::InvalidArgument("points/queries must be rank-2");
  }
  int num_points = static_cast<int>(points_dims[0]);
  int dim = static_cast<int>(points_dims[1]);
  int num_queries = static_cast<int>(queries_dims[0]);
  if (queries_dims[1] != dim) {
    return xla::ffi::Error::InvalidArgument("queries dim mismatch");
  }
  if (num_points != static_cast<int>(masses.element_count())) {
    return xla::ffi::Error::InvalidArgument("masses length mismatch");
  }
  if (dim != 2 && dim != 3) {
    return xla::ffi::Error::InvalidArgument("only dim=2 or dim=3 supported");
  }

  const float* points_ptr = points.typed_data();
  const float* masses_ptr = masses.typed_data();
  const float* queries_ptr = queries.typed_data();
  float* out_ptr = out->typed_data();

  int threads = 64;
  int blocks = (num_queries + threads - 1) / threads;
  brute_force_gravity_kernel<<<blocks, threads, 0, stream>>>(
      points_ptr, masses_ptr, queries_ptr, num_points, num_queries, dim,
      out_ptr);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(err));
  }

  return xla::ffi::Error::Success();
}

xla::ffi::Error BuildOctreeLevel3Buffers(cudaStream_t stream,
                                         F32R2 points,
                                         F32R2 normals,
                                         F32R1 masses,
                                         F32R1 min_corner,
                                         F32R1 max_corner,
                                         U8R1Out leaf_out,
                                         U8R1Out node_out,
                                         U8R1Out contrib_out) {
  if (points.dimensions().size() != 2 || points.dimensions()[1] != 3) {
    return xla::ffi::Error::InvalidArgument("points must have shape (N, 3)");
  }
  if (normals.dimensions().size() != 2 || normals.dimensions()[1] != 3) {
    return xla::ffi::Error::InvalidArgument("normals must have shape (N, 3)");
  }
  if (points.dimensions()[0] != normals.dimensions()[0]) {
    return xla::ffi::Error::InvalidArgument("points/normals row mismatch");
  }
  if (masses.dimensions().size() != 1 || masses.dimensions()[0] != points.dimensions()[0]) {
    return xla::ffi::Error::InvalidArgument("masses must have shape (N,)");
  }
  if (min_corner.dimensions().size() != 1 || min_corner.dimensions()[0] != 3 ||
      max_corner.dimensions().size() != 1 || max_corner.dimensions()[0] != 3) {
    return xla::ffi::Error::InvalidArgument("min_corner/max_corner must have shape (3,)");
  }

  int n = static_cast<int>(points.dimensions()[0]);
  size_t expected_leaf_bytes = static_cast<size_t>(n) * sizeof(Leaf3);
  size_t expected_node_bytes =
      static_cast<size_t>(kOctreeMaxNodes) * sizeof(Node3Layout);
  size_t expected_contrib_bytes =
      static_cast<size_t>(kOctreeMaxNodes) * sizeof(Leaf3);

  if (leaf_out->size_bytes() != expected_leaf_bytes) {
    return xla::ffi::Error::InvalidArgument("leaf_out size mismatch");
  }
  if (node_out->size_bytes() != expected_node_bytes) {
    return xla::ffi::Error::InvalidArgument("node_out size mismatch");
  }
  if (contrib_out->size_bytes() != expected_contrib_bytes) {
    return xla::ffi::Error::InvalidArgument("contrib_out size mismatch");
  }

  auto fail = [](cudaError_t err) {
    return xla::ffi::Error::Internal(cudaGetErrorString(err));
  };

  Leaf3* leaf_ptr = reinterpret_cast<Leaf3*>(leaf_out->typed_data());
  Node3Layout* node_ptr = reinterpret_cast<Node3Layout*>(node_out->typed_data());
  Leaf3* contrib_ptr = reinterpret_cast<Leaf3*>(contrib_out->typed_data());
  const float* points_ptr = points.typed_data();
  const float* normals_ptr = normals.typed_data();
  const float* masses_ptr = masses.typed_data();
  const float* min_ptr = min_corner.typed_data();
  const float* max_ptr = max_corner.typed_data();

  cudaError_t err = cudaMemsetAsync(leaf_ptr, 0, expected_leaf_bytes, stream);
  if (err != cudaSuccess) {
    return fail(err);
  }
  err = cudaMemsetAsync(node_ptr, 0, expected_node_bytes, stream);
  if (err != cudaSuccess) {
    return fail(err);
  }
  err = cudaMemsetAsync(contrib_ptr, 0, expected_contrib_bytes, stream);
  if (err != cudaSuccess) {
    return fail(err);
  }

  int* d_point_bins = nullptr;
  int* d_bin_counts = nullptr;
  int* d_bin_offsets = nullptr;
  int* d_bin_cursors = nullptr;
  int* d_bin_present = nullptr;
  int* d_child_ranks = nullptr;
  float* d_bin_mass_sums = nullptr;
  float* d_bin_pos_sums = nullptr;
  float* d_bin_normal_sums = nullptr;
  float* d_total_mass = nullptr;
  float* d_total_pos = nullptr;
  float* d_total_normal = nullptr;
  void* d_scan_temp = nullptr;

  auto cleanup = [&]() {
    cudaFree(d_point_bins);
    cudaFree(d_bin_counts);
    cudaFree(d_bin_offsets);
    cudaFree(d_bin_cursors);
    cudaFree(d_bin_present);
    cudaFree(d_child_ranks);
    cudaFree(d_bin_mass_sums);
    cudaFree(d_bin_pos_sums);
    cudaFree(d_bin_normal_sums);
    cudaFree(d_total_mass);
    cudaFree(d_total_pos);
    cudaFree(d_total_normal);
    cudaFree(d_scan_temp);
  };

  if (n > 0) {
    err = cudaMalloc(&d_point_bins, static_cast<size_t>(n) * sizeof(int));
    if (err != cudaSuccess) {
      cleanup();
      return fail(err);
    }
  }
  err = cudaMalloc(&d_bin_counts, static_cast<size_t>(kOctreeNumBins) * sizeof(int));
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMalloc(&d_bin_offsets, static_cast<size_t>(kOctreeNumBins) * sizeof(int));
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMalloc(&d_bin_cursors, static_cast<size_t>(kOctreeNumBins) * sizeof(int));
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMalloc(&d_bin_present, static_cast<size_t>(kOctreeNumBins) * sizeof(int));
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMalloc(&d_child_ranks, static_cast<size_t>(kOctreeNumBins) * sizeof(int));
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMalloc(&d_bin_mass_sums, static_cast<size_t>(kOctreeNumBins) * sizeof(float));
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMalloc(&d_bin_pos_sums, static_cast<size_t>(kOctreeNumBins) * 3 * sizeof(float));
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMalloc(&d_bin_normal_sums, static_cast<size_t>(kOctreeNumBins) * 3 * sizeof(float));
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMalloc(&d_total_mass, sizeof(float));
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMalloc(&d_total_pos, 3 * sizeof(float));
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMalloc(&d_total_normal, 3 * sizeof(float));
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }

  err = cudaMemsetAsync(d_bin_counts, 0,
                        static_cast<size_t>(kOctreeNumBins) * sizeof(int),
                        stream);
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMemsetAsync(d_bin_cursors, 0,
                        static_cast<size_t>(kOctreeNumBins) * sizeof(int),
                        stream);
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMemsetAsync(d_bin_present, 0,
                        static_cast<size_t>(kOctreeNumBins) * sizeof(int),
                        stream);
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMemsetAsync(d_bin_mass_sums, 0,
                        static_cast<size_t>(kOctreeNumBins) * sizeof(float),
                        stream);
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMemsetAsync(d_bin_pos_sums, 0,
                        static_cast<size_t>(kOctreeNumBins) * 3 * sizeof(float),
                        stream);
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMemsetAsync(d_bin_normal_sums, 0,
                        static_cast<size_t>(kOctreeNumBins) * 3 * sizeof(float),
                        stream);
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMemsetAsync(d_total_mass, 0, sizeof(float), stream);
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMemsetAsync(d_total_pos, 0, 3 * sizeof(float), stream);
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMemsetAsync(d_total_normal, 0, 3 * sizeof(float), stream);
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }

  int threads = 64;
  if (n > 0) {
    int blocks = (n + threads - 1) / threads;
    build_octree_level3_histogram_kernel<<<blocks, threads, 0, stream>>>(
        points_ptr, normals_ptr, masses_ptr, n, min_ptr, max_ptr, d_point_bins,
        d_bin_counts, d_bin_mass_sums, d_bin_pos_sums, d_bin_normal_sums,
        d_total_mass, d_total_pos, d_total_normal);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      cleanup();
      return fail(err);
    }
  }

  size_t scan_temp_bytes = 0;
  err = cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_bytes, d_bin_counts,
                                      d_bin_offsets, kOctreeNumBins, stream);
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMalloc(&d_scan_temp, scan_temp_bytes);
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cub::DeviceScan::ExclusiveSum(d_scan_temp, scan_temp_bytes, d_bin_counts,
                                      d_bin_offsets, kOctreeNumBins, stream);
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }

  if (n > 0) {
    int blocks = (n + threads - 1) / threads;
    build_octree_level3_scatter_leaf_kernel<<<blocks, threads, 0, stream>>>(
        points_ptr, normals_ptr, masses_ptr, n, d_point_bins, d_bin_offsets,
        d_bin_cursors, leaf_ptr);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      cleanup();
      return fail(err);
    }
  }

  int bin_blocks = (kOctreeNumBins + threads - 1) / threads;
  build_octree_level3_present_kernel<<<bin_blocks, threads, 0, stream>>>(
      d_bin_counts, d_bin_present);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }

  err = cub::DeviceScan::ExclusiveSum(d_scan_temp, scan_temp_bytes, d_bin_present,
                                      d_child_ranks, kOctreeNumBins, stream);
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }

  build_octree_level3_root_kernel<<<1, 1, 0, stream>>>(
      node_ptr, contrib_ptr, n, min_ptr, max_ptr, d_total_mass, d_total_pos,
      d_total_normal);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }

  build_octree_level3_children_kernel<<<bin_blocks, threads, 0, stream>>>(
      node_ptr, contrib_ptr, d_bin_counts, d_bin_offsets, d_child_ranks, min_ptr,
      max_ptr, d_bin_mass_sums, d_bin_pos_sums, d_bin_normal_sums);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }

  cleanup();
  return xla::ffi::Error::Success();
}

xla::ffi::Error BuildOctree8Buffers(cudaStream_t stream,
                                    F32R2 points,
                                    F32R2 normals,
                                    F32R1 masses,
                                    F32R1 min_corner,
                                    F32R1 max_corner,
                                    U8R1Out leaf_out,
                                    U8R1Out node_out,
                                    U8R1Out contrib_out,
                                    int max_depth) {
  if (points.dimensions().size() != 2 || points.dimensions()[1] != 3) {
    return xla::ffi::Error::InvalidArgument("points must have shape (N, 3)");
  }
  if (normals.dimensions().size() != 2 || normals.dimensions()[1] != 3) {
    return xla::ffi::Error::InvalidArgument("normals must have shape (N, 3)");
  }
  if (points.dimensions()[0] != normals.dimensions()[0]) {
    return xla::ffi::Error::InvalidArgument("points/normals row mismatch");
  }
  if (masses.dimensions().size() != 1 ||
      masses.dimensions()[0] != points.dimensions()[0]) {
    return xla::ffi::Error::InvalidArgument("masses must have shape (N,)");
  }
  if (min_corner.dimensions().size() != 1 ||
      min_corner.dimensions()[0] != 3 ||
      max_corner.dimensions().size() != 1 ||
      max_corner.dimensions()[0] != 3) {
    return xla::ffi::Error::InvalidArgument(
        "min_corner/max_corner must have shape (3,)");
  }
  if (max_depth < 1 || max_depth > 10) {
    return xla::ffi::Error::InvalidArgument("max_depth must be in [1, 10]");
  }

  int n = static_cast<int>(points.dimensions()[0]);
  uint32_t leaf_count = 1u << (3 * max_depth);
  uint32_t total_nodes = 0;
  uint32_t level_nodes = 1u;
  for (int l = 0; l <= max_depth; l++) {
    total_nodes += level_nodes;
    level_nodes <<= 3;
  }

  size_t expected_leaf_bytes = static_cast<size_t>(n) * sizeof(Leaf3);
  size_t expected_node_bytes =
      static_cast<size_t>(total_nodes) * sizeof(OctreeNode8);
  size_t expected_contrib_bytes =
      static_cast<size_t>(total_nodes) * sizeof(Leaf3);

  if (leaf_out->size_bytes() != expected_leaf_bytes) {
    return xla::ffi::Error::InvalidArgument("leaf_out size mismatch");
  }
  if (node_out->size_bytes() != expected_node_bytes) {
    return xla::ffi::Error::InvalidArgument("node_out size mismatch");
  }
  if (contrib_out->size_bytes() != expected_contrib_bytes) {
    return xla::ffi::Error::InvalidArgument("contrib_out size mismatch");
  }

  auto fail = [](cudaError_t err) {
    return xla::ffi::Error::Internal(cudaGetErrorString(err));
  };

  Leaf3* leaf_ptr = reinterpret_cast<Leaf3*>(leaf_out->typed_data());
  OctreeNode8* node_ptr = reinterpret_cast<OctreeNode8*>(node_out->typed_data());
  Leaf3* contrib_ptr = reinterpret_cast<Leaf3*>(contrib_out->typed_data());
  const float* points_ptr = points.typed_data();
  const float* normals_ptr = normals.typed_data();
  const float* masses_ptr = masses.typed_data();
  const float* min_ptr = min_corner.typed_data();
  const float* max_ptr = max_corner.typed_data();

  float h_min[3];
  float h_max[3];
  cudaError_t err = cudaMemcpyAsync(h_min, min_ptr, 3 * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return fail(err);
  }
  err = cudaMemcpyAsync(h_max, max_ptr, 3 * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return fail(err);
  }
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return fail(err);
  }

  float span_h = fmaxf(fmaxf(h_max[0] - h_min[0],
                             h_max[1] - h_min[1]),
                       h_max[2] - h_min[2]);
  if (span_h <= 0.0f) {
    return xla::ffi::Error::InvalidArgument("invalid bounds span");
  }

  err = cudaMemsetAsync(leaf_ptr, 0, expected_leaf_bytes, stream);
  if (err != cudaSuccess) {
    return fail(err);
  }
  err = cudaMemsetAsync(node_ptr, 0, expected_node_bytes, stream);
  if (err != cudaSuccess) {
    return fail(err);
  }
  err = cudaMemsetAsync(contrib_ptr, 0, expected_contrib_bytes, stream);
  if (err != cudaSuccess) {
    return fail(err);
  }

  uint32_t* d_leaf_codes = nullptr;
  uint32_t* d_leaf_counts = nullptr;
  uint32_t* d_leaf_offsets = nullptr;
  uint32_t* d_leaf_cursors = nullptr;
  float* d_leaf_mass_sums = nullptr;
  float* d_leaf_pos_sums = nullptr;
  float* d_leaf_norm_sums = nullptr;
  float* d_total_mass = nullptr;
  float* d_total_pos = nullptr;
  float* d_total_norm = nullptr;
  void* d_scan_temp = nullptr;

  std::vector<uint32_t*> level_counts;
  std::vector<float*> level_mass_sums;
  std::vector<float*> level_pos_sums;
  std::vector<float*> level_norm_sums;

  auto cleanup = [&]() {
    cudaFree(d_leaf_codes);
    cudaFree(d_leaf_offsets);
    cudaFree(d_leaf_cursors);
    cudaFree(d_total_mass);
    cudaFree(d_total_pos);
    cudaFree(d_total_norm);
    cudaFree(d_scan_temp);
    for (auto p : level_counts) cudaFree(p);
    for (auto p : level_mass_sums) cudaFree(p);
    for (auto p : level_pos_sums) cudaFree(p);
    for (auto p : level_norm_sums) cudaFree(p);
  };

  if (n > 0) {
    err = cudaMalloc(&d_leaf_codes, static_cast<size_t>(n) * sizeof(uint32_t));
    if (err != cudaSuccess) {
      cleanup();
      return fail(err);
    }
  }
  err = cudaMalloc(&d_leaf_counts,
                   static_cast<size_t>(leaf_count) * sizeof(uint32_t));
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMalloc(&d_leaf_offsets,
                   static_cast<size_t>(leaf_count) * sizeof(uint32_t));
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMalloc(&d_leaf_cursors,
                   static_cast<size_t>(leaf_count) * sizeof(uint32_t));
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMalloc(&d_leaf_mass_sums,
                   static_cast<size_t>(leaf_count) * sizeof(float));
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMalloc(&d_leaf_pos_sums,
                   static_cast<size_t>(leaf_count) * 3 * sizeof(float));
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMalloc(&d_leaf_norm_sums,
                   static_cast<size_t>(leaf_count) * 3 * sizeof(float));
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMalloc(&d_total_mass, sizeof(float));
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMalloc(&d_total_pos, 3 * sizeof(float));
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMalloc(&d_total_norm, 3 * sizeof(float));
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }

  err = cudaMemsetAsync(d_leaf_counts, 0,
                        static_cast<size_t>(leaf_count) * sizeof(uint32_t),
                        stream);
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMemsetAsync(d_leaf_cursors, 0,
                        static_cast<size_t>(leaf_count) * sizeof(uint32_t),
                        stream);
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMemsetAsync(d_leaf_mass_sums, 0,
                        static_cast<size_t>(leaf_count) * sizeof(float),
                        stream);
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMemsetAsync(d_leaf_pos_sums, 0,
                        static_cast<size_t>(leaf_count) * 3 * sizeof(float),
                        stream);
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMemsetAsync(d_leaf_norm_sums, 0,
                        static_cast<size_t>(leaf_count) * 3 * sizeof(float),
                        stream);
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMemsetAsync(d_total_mass, 0, sizeof(float), stream);
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMemsetAsync(d_total_pos, 0, 3 * sizeof(float), stream);
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMemsetAsync(d_total_norm, 0, 3 * sizeof(float), stream);
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }

  int threads = 64;
  if (n > 0) {
    int blocks = (n + threads - 1) / threads;
    octree8_histogram_kernel<<<blocks, threads, 0, stream>>>(
        points_ptr, normals_ptr, masses_ptr, n, min_ptr, span_h, max_depth,
        d_leaf_codes, d_leaf_counts, d_leaf_mass_sums, d_leaf_pos_sums,
        d_leaf_norm_sums, d_total_mass, d_total_pos, d_total_norm);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      cleanup();
      return fail(err);
    }
  }

  size_t scan_temp_bytes = 0;
  err = cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_bytes, d_leaf_counts,
                                      d_leaf_offsets, leaf_count, stream);
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cudaMalloc(&d_scan_temp, scan_temp_bytes);
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }
  err = cub::DeviceScan::ExclusiveSum(d_scan_temp, scan_temp_bytes,
                                      d_leaf_counts, d_leaf_offsets, leaf_count,
                                      stream);
  if (err != cudaSuccess) {
    cleanup();
    return fail(err);
  }

  if (n > 0) {
    int blocks = (n + threads - 1) / threads;
    octree8_scatter_leaf_kernel<<<blocks, threads, 0, stream>>>(
        points_ptr, normals_ptr, masses_ptr, n, d_leaf_codes, d_leaf_offsets,
        d_leaf_cursors, leaf_ptr);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      cleanup();
      return fail(err);
    }
  }

  level_counts.resize(max_depth + 1);
  level_mass_sums.resize(max_depth + 1);
  level_pos_sums.resize(max_depth + 1);
  level_norm_sums.resize(max_depth + 1);

  level_counts[max_depth] = d_leaf_counts;
  level_mass_sums[max_depth] = d_leaf_mass_sums;
  level_pos_sums[max_depth] = d_leaf_pos_sums;
  level_norm_sums[max_depth] = d_leaf_norm_sums;

  for (int l = max_depth - 1; l >= 0; l--) {
    uint32_t count = 1u << (3 * l);
    float* mass = nullptr;
    float* pos = nullptr;
    float* norm = nullptr;
    uint32_t* cnt = nullptr;
    err = cudaMalloc(&cnt, static_cast<size_t>(count) * sizeof(uint32_t));
    if (err != cudaSuccess) {
      cleanup();
      return fail(err);
    }
    err = cudaMalloc(&mass, static_cast<size_t>(count) * sizeof(float));
    if (err != cudaSuccess) {
      cleanup();
      return fail(err);
    }
    err = cudaMalloc(&pos, static_cast<size_t>(count) * 3 * sizeof(float));
    if (err != cudaSuccess) {
      cleanup();
      return fail(err);
    }
    err = cudaMalloc(&norm, static_cast<size_t>(count) * 3 * sizeof(float));
    if (err != cudaSuccess) {
      cleanup();
      return fail(err);
    }
    level_counts[l] = cnt;
    level_mass_sums[l] = mass;
    level_pos_sums[l] = pos;
    level_norm_sums[l] = norm;

    int blocks = (count + threads - 1) / threads;
    octree8_reduce_level_kernel<<<blocks, threads, 0, stream>>>(
        level_counts[l + 1], level_mass_sums[l + 1], level_pos_sums[l + 1],
        level_norm_sums[l + 1], level_counts[l], level_mass_sums[l],
        level_pos_sums[l], level_norm_sums[l], count);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      cleanup();
      return fail(err);
    }
  }

  std::vector<uint32_t> level_offsets(max_depth + 1, 0);
  uint32_t offset = 0;
  for (int l = 0; l <= max_depth; l++) {
    level_offsets[l] = offset;
    offset += 1u << (3 * l);
  }

  float cx = h_min[0] + span_h * 0.5f;
  float cy = h_min[1] + span_h * 0.5f;
  float cz = h_min[2] + span_h * 0.5f;
  float half_size = span_h * 0.5f;

  for (int l = 0; l <= max_depth; l++) {
    uint32_t count = 1u << (3 * l);
    int blocks = (count + threads - 1) / threads;
    const uint32_t* child_counts =
        (l < max_depth) ? level_counts[l + 1] : nullptr;
    octree8_build_nodes_kernel<<<blocks, threads, 0, stream>>>(
        node_ptr, contrib_ptr, d_leaf_offsets, level_counts[l],
        level_mass_sums[l], level_pos_sums[l], level_norm_sums[l],
        child_counts, l, max_depth, level_offsets[l],
        l == 0 ? 0u : level_offsets[l - 1], l < max_depth ? level_offsets[l + 1] : 0u,
        cx, cy, cz, half_size);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      cleanup();
      return fail(err);
    }
  }

  cleanup();
  return xla::ffi::Error::Success();
}

__global__ void barnes_hut_gravity_kernel(const Leaf2* leaf_data,
                                          const Node2* nodes,
                                          const float* queries,
                                          int num_queries,
                                          float beta,
                                          float* out) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_queries) {
    return;
  }

  VectorNd<2> qv;
  qv(0) = queries[q * 2 + 0];
  qv(1) = queries[q * 2 + 1];

  out[q] = evaluateBarnesHutGPUVote<FLOAT, Node2>(
      leaf_data, nodes, nodes, qv, beta, gravityPotential);
}

__global__ void barnes_hut_gravity_kernel_3d(const Leaf3* leaf_data,
                                             const Node3* nodes,
                                             const float* queries,
                                             int num_queries,
                                             float beta,
                                             float* out) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_queries) {
    return;
  }

  VectorNd<3> qv;
  qv(0) = queries[q * 3 + 0];
  qv(1) = queries[q * 3 + 1];
  qv(2) = queries[q * 3 + 2];

  out[q] = evaluateBarnesHutGPUVote<FLOAT, Node3>(
      leaf_data, nodes, nodes, qv, beta, gravityPotential);
}

__global__ void barnes_hut_force_kernel(const Leaf2* leaf_data,
                                        const Node2* nodes,
                                        const float* queries,
                                        int num_queries,
                                        float beta,
                                        float* out) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_queries) {
    return;
  }

  VectorNd<2> qv;
  qv(0) = queries[q * 2 + 0];
  qv(1) = queries[q * 2 + 1];

  VectorNd<2> f = evaluateBarnesHutGPUVote<VectorNd<2>, Node2>(
      leaf_data, nodes, nodes, qv, beta, gravityGrad);

  out[q * 2 + 0] = f(0);
  out[q * 2 + 1] = f(1);
}

__global__ void barnes_hut_force_kernel_3d(const Leaf3* leaf_data,
                                           const Node3* nodes,
                                           const float* queries,
                                           int num_queries,
                                           float beta,
                                           float* out) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_queries) {
    return;
  }

  VectorNd<3> qv;
  qv(0) = queries[q * 3 + 0];
  qv(1) = queries[q * 3 + 1];
  qv(2) = queries[q * 3 + 2];

  VectorNd<3> f = evaluateBarnesHutGPUVote<VectorNd<3>, Node3>(
      leaf_data, nodes, nodes, qv, beta, gravityGrad);

  out[q * 3 + 0] = f(0);
  out[q * 3 + 1] = f(1);
  out[q * 3 + 2] = f(2);
}

__global__ void barnes_hut_force_soft_kernel(const Leaf2* leaf_data,
                                             const Node2* nodes,
                                             const float* queries,
                                             int num_queries,
                                             float beta,
                                             float* out) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_queries) {
    return;
  }

  VectorNd<2> qv;
  qv(0) = queries[q * 2 + 0];
  qv(1) = queries[q * 2 + 1];

  VectorNd<2> f = evaluateBarnesHutGPUVote<VectorNd<2>, Node2>(
      leaf_data, nodes, nodes, qv, beta, gravityGradSoftCutoff<2>);

  out[q * 2 + 0] = f(0);
  out[q * 2 + 1] = f(1);
}

__global__ void barnes_hut_force_soft_kernel_3d(const Leaf3* leaf_data,
                                                const Node3* nodes,
                                                const float* queries,
                                                int num_queries,
                                                float beta,
                                                float* out) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_queries) {
    return;
  }

  VectorNd<3> qv;
  qv(0) = queries[q * 3 + 0];
  qv(1) = queries[q * 3 + 1];
  qv(2) = queries[q * 3 + 2];

  VectorNd<3> f = evaluateBarnesHutGPUVote<VectorNd<3>, Node3>(
      leaf_data, nodes, nodes, qv, beta, gravityGradSoftCutoff<3>);

  out[q * 3 + 0] = f(0);
  out[q * 3 + 1] = f(1);
  out[q * 3 + 2] = f(2);
}

__global__ void stochastic_bh_gravity_kernel_2d(const Leaf2* leaf_data,
                                                int num_points,
                                                const Node2* nodes,
                                                const Leaf2* contrib,
                                                const float* queries,
                                                int num_queries,
                                                int samples_per_subdomain,
                                                int seed,
                                                float* out) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_queries) return;
  VectorNd<2> qv;
  qv(0) = queries[q * 2 + 0];
  qv(1) = queries[q * 2 + 1];
  out[q] = multiLevelPrefixControlVariateSampleGPUQuery<FLOAT, Node2>(
      leaf_data, num_points, nodes, nodes, contrib, qv, gravityPotential,
      samples_per_subdomain, seed + q);
}

__global__ void stochastic_bh_gravity_kernel_3d(const Leaf3* leaf_data,
                                                int num_points,
                                                const Node3* nodes,
                                                const Leaf3* contrib,
                                                const float* queries,
                                                int num_queries,
                                                int samples_per_subdomain,
                                                int seed,
                                                float* out) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_queries) return;
  VectorNd<3> qv;
  qv(0) = queries[q * 3 + 0];
  qv(1) = queries[q * 3 + 1];
  qv(2) = queries[q * 3 + 2];
  out[q] = multiLevelPrefixControlVariateSampleGPUQuery<FLOAT, Node3>(
      leaf_data, num_points, nodes, nodes, contrib, qv, gravityPotential,
      samples_per_subdomain, seed + q);
}

__global__ void stochastic_bh_force_kernel_2d(const Leaf2* leaf_data,
                                              int num_points,
                                              const Node2* nodes,
                                              const Leaf2* contrib,
                                              const float* queries,
                                              int num_queries,
                                              int samples_per_subdomain,
                                              int seed,
                                              float* out) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_queries) return;
  VectorNd<2> qv;
  qv(0) = queries[q * 2 + 0];
  qv(1) = queries[q * 2 + 1];
  VectorNd<2> f = multiLevelPrefixControlVariateSampleGPUQuery<VectorNd<2>, Node2>(
      leaf_data, num_points, nodes, nodes, contrib, qv, gravityGrad,
      samples_per_subdomain, seed + q);
  out[q * 2 + 0] = f(0);
  out[q * 2 + 1] = f(1);
}

__global__ void stochastic_bh_force_kernel_3d(const Leaf3* leaf_data,
                                              int num_points,
                                              const Node3* nodes,
                                              const Leaf3* contrib,
                                              const float* queries,
                                              int num_queries,
                                              int samples_per_subdomain,
                                              int seed,
                                              float* out) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_queries) return;
  VectorNd<3> qv;
  qv(0) = queries[q * 3 + 0];
  qv(1) = queries[q * 3 + 1];
  qv(2) = queries[q * 3 + 2];
  VectorNd<3> f = multiLevelPrefixControlVariateSampleGPUQuery<VectorNd<3>, Node3>(
      leaf_data, num_points, nodes, nodes, contrib, qv, gravityGrad,
      samples_per_subdomain, seed + q);
  out[q * 3 + 0] = f(0);
  out[q * 3 + 1] = f(1);
  out[q * 3 + 2] = f(2);
}

__global__ void stochastic_bh_force_soft_kernel_2d(const Leaf2* leaf_data,
                                                   int num_points,
                                                   const Node2* nodes,
                                                   const Leaf2* contrib,
                                                   const float* queries,
                                                   int num_queries,
                                                   int samples_per_subdomain,
                                                   int seed,
                                                   float* out) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_queries) return;
  VectorNd<2> qv;
  qv(0) = queries[q * 2 + 0];
  qv(1) = queries[q * 2 + 1];
  VectorNd<2> f =
      multiLevelPrefixControlVariateSampleGPUQuery<VectorNd<2>, Node2>(
          leaf_data, num_points, nodes, nodes, contrib, qv,
          gravityGradSoftCutoff<2>, samples_per_subdomain, seed + q);
  out[q * 2 + 0] = f(0);
  out[q * 2 + 1] = f(1);
}

__global__ void stochastic_bh_force_soft_kernel_3d(const Leaf3* leaf_data,
                                                   int num_points,
                                                   const Node3* nodes,
                                                   const Leaf3* contrib,
                                                   const float* queries,
                                                   int num_queries,
                                                   int samples_per_subdomain,
                                                   int seed,
                                                   float* out) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_queries) return;
  VectorNd<3> qv;
  qv(0) = queries[q * 3 + 0];
  qv(1) = queries[q * 3 + 1];
  qv(2) = queries[q * 3 + 2];
  VectorNd<3> f =
      multiLevelPrefixControlVariateSampleGPUQuery<VectorNd<3>, Node3>(
          leaf_data, num_points, nodes, nodes, contrib, qv,
          gravityGradSoftCutoff<3>, samples_per_subdomain, seed + q);
  out[q * 3 + 0] = f(0);
  out[q * 3 + 1] = f(1);
  out[q * 3 + 2] = f(2);
}

__global__ void octree8_histogram_kernel(
    const float* points, const float* normals, const float* masses, int n,
    const float* min_corner, float span, int depth_bits, uint32_t* leaf_codes,
    uint32_t* leaf_counts, float* leaf_mass_sums, float* leaf_pos_sums,
    float* leaf_normal_sums, float* total_mass, float* total_pos,
    float* total_normal) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  float px = points[i * 3 + 0];
  float py = points[i * 3 + 1];
  float pz = points[i * 3 + 2];
  float nx = normals[i * 3 + 0];
  float ny = normals[i * 3 + 1];
  float nz = normals[i * 3 + 2];
  float m = masses[i];

  uint32_t qx = quantize_axis(px, min_corner[0], span, depth_bits);
  uint32_t qy = quantize_axis(py, min_corner[1], span, depth_bits);
  uint32_t qz = quantize_axis(pz, min_corner[2], span, depth_bits);
  uint32_t code = morton3d(qx, qy, qz, depth_bits);

  leaf_codes[i] = code;
  atomicAdd(reinterpret_cast<unsigned int*>(&leaf_counts[code]), 1u);
  atomicAdd(&leaf_mass_sums[code], m);
  atomicAdd(&leaf_pos_sums[code * 3 + 0], m * px);
  atomicAdd(&leaf_pos_sums[code * 3 + 1], m * py);
  atomicAdd(&leaf_pos_sums[code * 3 + 2], m * pz);
  atomicAdd(&leaf_normal_sums[code * 3 + 0], m * nx);
  atomicAdd(&leaf_normal_sums[code * 3 + 1], m * ny);
  atomicAdd(&leaf_normal_sums[code * 3 + 2], m * nz);
  atomicAdd(&total_mass[0], m);
  atomicAdd(&total_pos[0], m * px);
  atomicAdd(&total_pos[1], m * py);
  atomicAdd(&total_pos[2], m * pz);
  atomicAdd(&total_normal[0], m * nx);
  atomicAdd(&total_normal[1], m * ny);
  atomicAdd(&total_normal[2], m * nz);
}

__global__ void octree8_scatter_leaf_kernel(
    const float* points, const float* normals, const float* masses, int n,
    const uint32_t* leaf_codes, const uint32_t* leaf_offsets,
    uint32_t* leaf_cursors, Leaf3* leaf_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  uint32_t code = leaf_codes[i];
  uint32_t dst = leaf_offsets[code] + atomicAdd(&leaf_cursors[code], 1u);
  Leaf3 leaf;
  leaf.position(0) = points[i * 3 + 0];
  leaf.position(1) = points[i * 3 + 1];
  leaf.position(2) = points[i * 3 + 2];
  leaf.normal(0) = normals[i * 3 + 0];
  leaf.normal(1) = normals[i * 3 + 1];
  leaf.normal(2) = normals[i * 3 + 2];
  leaf.mass = masses[i];
  leaf.alpha = 1.0f;
  leaf_out[dst] = leaf;
}

__global__ void octree8_reduce_level_kernel(
    const uint32_t* child_counts, const float* child_mass_sums,
    const float* child_pos_sums, const float* child_normal_sums,
    uint32_t* parent_counts, float* parent_mass_sums, float* parent_pos_sums,
    float* parent_normal_sums, int parent_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= parent_count) {
    return;
  }
  uint32_t base = static_cast<uint32_t>(idx) * 8u;
  uint32_t count = 0;
  float mass = 0.0f;
  float pos_x = 0.0f;
  float pos_y = 0.0f;
  float pos_z = 0.0f;
  float norm_x = 0.0f;
  float norm_y = 0.0f;
  float norm_z = 0.0f;
  for (uint32_t c = 0; c < 8; c++) {
    uint32_t child = base + c;
    uint32_t cc = child_counts[child];
    if (cc == 0) {
      continue;
    }
    count += cc;
    mass += child_mass_sums[child];
    pos_x += child_pos_sums[child * 3 + 0];
    pos_y += child_pos_sums[child * 3 + 1];
    pos_z += child_pos_sums[child * 3 + 2];
    norm_x += child_normal_sums[child * 3 + 0];
    norm_y += child_normal_sums[child * 3 + 1];
    norm_z += child_normal_sums[child * 3 + 2];
  }
  parent_counts[idx] = count;
  parent_mass_sums[idx] = mass;
  parent_pos_sums[idx * 3 + 0] = pos_x;
  parent_pos_sums[idx * 3 + 1] = pos_y;
  parent_pos_sums[idx * 3 + 2] = pos_z;
  parent_normal_sums[idx * 3 + 0] = norm_x;
  parent_normal_sums[idx * 3 + 1] = norm_y;
  parent_normal_sums[idx * 3 + 2] = norm_z;
}

__global__ void octree8_build_nodes_kernel(
    OctreeNode8* nodes, Leaf3* contrib_out, const uint32_t* leaf_offsets,
    const uint32_t* level_counts, const float* level_mass_sums,
    const float* level_pos_sums, const float* level_normal_sums,
    const uint32_t* child_counts, int level, int max_level,
    uint32_t level_offset, uint32_t parent_level_offset,
    uint32_t child_level_offset, float cx, float cy, float cz,
    float half_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t level_nodes = 1u << (3 * level);
  if (idx >= level_nodes) {
    return;
  }
  uint32_t node_idx = level_offset + static_cast<uint32_t>(idx);
  uint32_t count = level_counts[idx];
  OctreeNode8 node{};

  node.depth = static_cast<uint8_t>(level);
  node.parent = (level == 0)
                    ? node_idx
                    : (parent_level_offset + (static_cast<uint32_t>(idx) >> 3));
  node.is_leaf = (level == max_level) ? 1 : 0;
  node.child_mask = 0;
  node.first_child = 0xFFFFFFFFu;

  uint32_t prefix = static_cast<uint32_t>(idx);
  int shift = 3 * (max_level - level);
  uint32_t first_leaf = prefix << shift;
  node.start = leaf_offsets[first_leaf];
  node.end = node.start + count;

  // center and size
  int grid = 1 << level;
  int ix = idx & (grid - 1);
  int iy = (idx >> level) & (grid - 1);
  int iz = (idx >> (2 * level)) & (grid - 1);
  float step = half_size * 2.0f / static_cast<float>(grid);
  node.cx = (cx - half_size) + (static_cast<float>(ix) + 0.5f) * step;
  node.cy = (cy - half_size) + (static_cast<float>(iy) + 0.5f) * step;
  node.cz = (cz - half_size) + (static_cast<float>(iz) + 0.5f) * step;
  node.half_size = step * 0.5f;
  node.diameter = step * sqrtf(3.0f);

  if (level < max_level) {
    uint32_t child_base = child_level_offset + (prefix << 3);
    uint8_t mask = 0;
    for (uint32_t c = 0; c < 8; c++) {
      uint32_t child_idx = child_base + c;
      node.child[c] = child_idx;
      if (child_counts[(prefix << 3) + c] > 0) {
        mask |= static_cast<uint8_t>(1u << c);
        if (node.first_child == 0xFFFFFFFFu) {
          node.first_child = child_idx;
        }
      }
    }
    node.child_mask = mask;
    if (mask == 0) {
      node.is_leaf = 1;
    }
  } else {
    for (uint32_t c = 0; c < 8; c++) {
      node.child[c] = 0xFFFFFFFFu;
    }
  }

  float mass = level_mass_sums[idx];
  node.contrib.alpha = 1.0f;
  node.contrib.mass = mass;
  if (mass > 0.0f) {
    node.contrib.position(0) = level_pos_sums[idx * 3 + 0] / mass;
    node.contrib.position(1) = level_pos_sums[idx * 3 + 1] / mass;
    node.contrib.position(2) = level_pos_sums[idx * 3 + 2] / mass;
    node.contrib.normal(0) = level_normal_sums[idx * 3 + 0] / mass;
    node.contrib.normal(1) = level_normal_sums[idx * 3 + 1] / mass;
    node.contrib.normal(2) = level_normal_sums[idx * 3 + 2] / mass;
  } else {
    node.contrib.position.setZero();
    node.contrib.normal.setZero();
  }

  nodes[node_idx] = node;
  contrib_out[node_idx] = node.contrib;
}

__device__ __forceinline__ bool octree8_use_far_field(
    const OctreeNode8& node, VectorNd<3> q, float beta) {
  if (node.is_leaf) {
    return true;
  }
  VectorNd<3> dc;
  dc(0) = q(0) - node.cx;
  dc(1) = q(1) - node.cy;
  dc(2) = q(2) - node.cz;
  dc = minImageDisplacement<3>(dc);
  float dx = dc(0);
  float dy = dc(1);
  float dz = dc(2);
  float dist = sqrtf(dx * dx + dy * dy + dz * dz);
  return dist / node.diameter >= beta;
}

__device__ __forceinline__ uint32_t octree8_child_slot(const OctreeNode8& node,
                                                       VectorNd<3> p) {
  uint32_t ix = p(0) >= node.cx ? 1u : 0u;
  uint32_t iy = p(1) >= node.cy ? 1u : 0u;
  uint32_t iz = p(2) >= node.cz ? 1u : 0u;
  return ix | (iy << 1) | (iz << 2);
}

__global__ void barnes_hut_force_octree8_kernel(
    const Leaf3* leaf_data, const OctreeNode8* nodes, const float* queries,
    int num_queries, float beta, float* out) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_queries) {
    return;
  }
  VectorNd<3> qv;
  qv(0) = queries[q * 3 + 0];
  qv(1) = queries[q * 3 + 1];
  qv(2) = queries[q * 3 + 2];

  VectorNd<3> acc = VectorNd<3>::Zero();
  int stack[128];
  int top = 0;
  stack[top++] = 0;
  while (top > 0) {
    int idx = stack[--top];
    const OctreeNode8& node = nodes[idx];
    if (node.contrib.mass == 0.0f) {
      continue;
    }
    if (octree8_use_far_field(node, qv, beta)) {
      if (node.is_leaf) {
        for (uint32_t i = node.start; i < node.end; i++) {
          acc += gravityGrad<3>(leaf_data[i], qv);
        }
      } else {
        acc += gravityGrad<3>(node.contrib, qv);
      }
    } else {
      uint8_t mask = node.child_mask;
      for (int c = 0; c < 8; c++) {
        if (mask & (1u << c)) {
          stack[top++] = node.child[c];
        }
      }
    }
  }

  out[q * 3 + 0] = acc(0);
  out[q * 3 + 1] = acc(1);
  out[q * 3 + 2] = acc(2);
}

__global__ void barnes_hut_force_octree8_soft_kernel(
    const Leaf3* leaf_data, const OctreeNode8* nodes, const float* queries,
    int num_queries, float beta, float* out) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_queries) {
    return;
  }
  VectorNd<3> qv;
  qv(0) = queries[q * 3 + 0];
  qv(1) = queries[q * 3 + 1];
  qv(2) = queries[q * 3 + 2];

  VectorNd<3> acc = VectorNd<3>::Zero();
  int stack[128];
  int top = 0;
  stack[top++] = 0;
  while (top > 0) {
    int idx = stack[--top];
    const OctreeNode8& node = nodes[idx];
    if (node.contrib.mass == 0.0f) {
      continue;
    }
    if (octree8_use_far_field(node, qv, beta)) {
      if (node.is_leaf) {
        for (uint32_t i = node.start; i < node.end; i++) {
          acc += gravityGradSoftCutoff<3>(leaf_data[i], qv);
        }
      } else {
        acc += gravityGradSoftCutoff<3>(node.contrib, qv);
      }
    } else {
      uint8_t mask = node.child_mask;
      for (int c = 0; c < 8; c++) {
        if (mask & (1u << c)) {
          stack[top++] = node.child[c];
        }
      }
    }
  }

  out[q * 3 + 0] = acc(0);
  out[q * 3 + 1] = acc(1);
  out[q * 3 + 2] = acc(2);
}

__device__ __forceinline__ VectorNd<3> gravityGradSoftCutoffVjp(
    const Leaf3& data, VectorNd<3> q, VectorNd<3> g) {
  VectorNd<3> r = minImageDisplacement<3>(q - data.position);
  float r2 = r.squaredNorm();
  float eps = fmaxf(g_softening_scale, 0.0f);
  float inv_r = rsqrtf(r2 + eps * eps + 1e-12f);
  float inv_r3 = inv_r * inv_r * inv_r;
  float inv_r5 = inv_r3 * inv_r * inv_r;
  float attenuation = 1.0f;
  float cutoff_coeff = 0.0f;
  if (g_cutoff_scale > 0.0f) {
    float dist = sqrtf(r2 + 1e-12f);
    float u = dist / (2.0f * g_cutoff_scale);
    float exp_u2 = expf(-(u * u));
    attenuation = erfcf(u) + (2.0f / 1.772453850905516f) * u * exp_u2;
    // For J^T g of f(r)= -m * attenuation(r) * inv_r3 * r:
    // use coefficient 3*attenuation*inv_r5 - inv_r3 * attenuation'(r)/r.
    // Here attenuation'(r)/r = -exp(-u^2) / (2 sqrt(pi) r_split^3) * r.
    cutoff_coeff =
        inv_r3 * dist * exp_u2 /
        (2.0f * 1.772453850905516f * g_cutoff_scale * g_cutoff_scale *
         g_cutoff_scale);
  }
  float rg = r.dot(g);
  VectorNd<3> grad = attenuation * inv_r3 * g -
                     rg * (3.0f * attenuation * inv_r5 + cutoff_coeff) * r;
  return -data.mass * grad;
}

__global__ void barnes_hut_force_octree8_soft_vjp_kernel(
    const Leaf3* leaf_data, const OctreeNode8* nodes, const float* queries,
    const float* cotangent, int num_queries, float beta, float* out) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_queries) {
    return;
  }
  VectorNd<3> qv;
  qv(0) = queries[q * 3 + 0];
  qv(1) = queries[q * 3 + 1];
  qv(2) = queries[q * 3 + 2];

  VectorNd<3> gv;
  gv(0) = cotangent[q * 3 + 0];
  gv(1) = cotangent[q * 3 + 1];
  gv(2) = cotangent[q * 3 + 2];

  VectorNd<3> grad = VectorNd<3>::Zero();
  int stack[128];
  int top = 0;
  stack[top++] = 0;
  while (top > 0) {
    int idx = stack[--top];
    const OctreeNode8& node = nodes[idx];
    if (node.contrib.mass == 0.0f) {
      continue;
    }
    if (octree8_use_far_field(node, qv, beta)) {
      if (node.is_leaf) {
        for (uint32_t i = node.start; i < node.end; i++) {
          grad += gravityGradSoftCutoffVjp(leaf_data[i], qv, gv);
        }
      } else {
        grad += gravityGradSoftCutoffVjp(node.contrib, qv, gv);
      }
    } else {
      uint8_t mask = node.child_mask;
      for (int c = 0; c < 8; c++) {
        if (mask & (1u << c)) {
          stack[top++] = node.child[c];
        }
      }
    }
  }

  out[q * 3 + 0] = grad(0);
  out[q * 3 + 1] = grad(1);
  out[q * 3 + 2] = grad(2);
}

__global__ void stochastic_bh_force_octree8_kernel(
    const Leaf3* leaf_data, const OctreeNode8* nodes, const float* queries,
    int num_queries, int samples_per_subdomain, int seed, float* out,
    bool softened) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_queries) {
    return;
  }
  VectorNd<3> qv;
  qv(0) = queries[q * 3 + 0];
  qv(1) = queries[q * 3 + 1];
  qv(2) = queries[q * 3 + 2];

  const OctreeNode8& root = nodes[0];
  VectorNd<3> total = VectorNd<3>::Zero();
  curandState s;
  curandState s2;
  curand_init(seed + q, 0, 0, &s);
  curand_init(blockIdx.x, 0, 0, &s2);

  uint8_t root_mask = root.child_mask;
  for (int c = 0; c < 8; c++) {
    if (!(root_mask & (1u << c))) {
      continue;
    }
    const OctreeNode8& sub = nodes[root.child[c]];
    if (sub.is_leaf) {
      for (uint32_t i = sub.start; i < sub.end; i++) {
        total += softened ? gravityGradSoftCutoff<3>(leaf_data[i], qv)
                          : gravityGrad<3>(leaf_data[i], qv);
      }
    } else {
      total += softened ? gravityGradSoftCutoff<3>(sub.contrib, qv)
                        : gravityGrad<3>(sub.contrib, qv);
    }

    uint32_t num_elements = sub.end - sub.start;
    VectorNd<3> sub_total = VectorNd<3>::Zero();
    for (int ss = 0; ss < samples_per_subdomain; ss++) {
      uint32_t idx =
          min(static_cast<uint32_t>(num_elements * curand_uniform(&s2)) +
                  sub.start,
              sub.end - 1);
      VectorNd<3> sample_pos = leaf_data[idx].position;

      const OctreeNode8* prev_node = &sub;
      const OctreeNode8* cur_node = &sub;
      float event_prob = 1.0f;
      VectorNd<3> weighted_diff_all = VectorNd<3>::Zero();
      VectorNd<3> dcur = qv - VectorNd<3>(cur_node->cx, cur_node->cy, cur_node->cz);
      dcur = minImageDisplacement<3>(dcur);
      float cur_beta = fmaxf(1.0f, (float)(dcur.norm() / cur_node->diameter));
      while (!cur_node->is_leaf) {
        VectorNd<3> dnode = qv - VectorNd<3>(cur_node->cx, cur_node->cy, cur_node->cz);
        dnode = minImageDisplacement<3>(dnode);
        float dist = dnode.norm();
        float far_field_ratio = dist / cur_node->diameter;
        float traverse_prob = fminf(cur_beta / far_field_ratio, 1.0f);
        float p = curand_uniform(&s);
        if (p > traverse_prob) {
          event_prob *= 1.0f - traverse_prob;
          break;
        }
        event_prob *= traverse_prob;
        prev_node = cur_node;
        uint32_t slot = octree8_child_slot(*cur_node, sample_pos);
        if (!(cur_node->child_mask & (1u << slot))) {
          break;
        }
        cur_node = &nodes[cur_node->child[slot]];
        cur_beta = fmaxf(cur_beta, far_field_ratio);

        VectorNd<3> prev_result =
            softened ? gravityGradSoftCutoff<3>(prev_node->contrib, qv)
                     : gravityGrad<3>(prev_node->contrib, qv);
        VectorNd<3> cur_result = VectorNd<3>::Zero();
        uint8_t mask = prev_node->child_mask;
        for (int cc = 0; cc < 8; cc++) {
          if (!(mask & (1u << cc))) {
            continue;
          }
          const OctreeNode8& child = nodes[prev_node->child[cc]];
          if (child.is_leaf) {
            for (uint32_t i = child.start; i < child.end; i++) {
              cur_result += softened
                                 ? gravityGradSoftCutoff<3>(leaf_data[i], qv)
                                 : gravityGrad<3>(leaf_data[i], qv);
            }
          } else {
            cur_result +=
                softened ? gravityGradSoftCutoff<3>(child.contrib, qv)
                         : gravityGrad<3>(child.contrib, qv);
          }
        }
        VectorNd<3> diff = cur_result - prev_result;
        float prob_inv =
            static_cast<float>(num_elements) /
            (event_prob * static_cast<float>(prev_node->end - prev_node->start));
        weighted_diff_all += prob_inv * diff;
      }
      sub_total += weighted_diff_all;
    }
    if (samples_per_subdomain > 0) {
      total += sub_total / static_cast<float>(samples_per_subdomain);
    }
  }

  out[q * 3 + 0] = total(0);
  out[q * 3 + 1] = total(1);
  out[q * 3 + 2] = total(2);
}

xla::ffi::Error BarnesHutGravity(cudaStream_t stream,
                                 U8R1 leaf_bytes,
                                 U8R1 node_bytes,
                                 F32R2 queries,
                                 F32R1Out out,
                                 float beta,
                                 int periodic,
                                 float box_length) {
  if (leaf_bytes.size_bytes() % sizeof(Leaf2) != 0) {
    return xla::ffi::Error::InvalidArgument("leaf_bytes size mismatch");
  }
  if (node_bytes.size_bytes() % sizeof(Node2) != 0) {
    return xla::ffi::Error::InvalidArgument("node_bytes size mismatch");
  }

  int num_queries = static_cast<int>(queries.dimensions()[0]);
  if (queries.dimensions()[1] != 2) {
    return xla::ffi::Error::InvalidArgument("queries dim must be 2");
  }

  cudaError_t per_err = SetPeriodicBox(stream, periodic, box_length);
  if (per_err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(per_err));
  }

  const Leaf2* leaf_data = reinterpret_cast<const Leaf2*>(leaf_bytes.untyped_data());
  const Node2* nodes = reinterpret_cast<const Node2*>(node_bytes.untyped_data());
  const float* queries_ptr = queries.typed_data();
  float* out_ptr = out->typed_data();

  int threads = 64;
  int blocks = (num_queries + threads - 1) / threads;
  barnes_hut_gravity_kernel<<<blocks, threads, 0, stream>>>(
      leaf_data, nodes, queries_ptr, num_queries, beta, out_ptr);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(err));
  }

  return xla::ffi::Error::Success();
}

xla::ffi::Error BarnesHutGravity3D(cudaStream_t stream,
                                   U8R1 leaf_bytes,
                                   U8R1 node_bytes,
                                   F32R2 queries,
                                   F32R1Out out,
                                   float beta,
                                   int periodic,
                                   float box_length) {
  if (leaf_bytes.size_bytes() % sizeof(Leaf3) != 0) {
    return xla::ffi::Error::InvalidArgument("leaf_bytes size mismatch");
  }
  if (node_bytes.size_bytes() % sizeof(Node3) != 0) {
    return xla::ffi::Error::InvalidArgument("node_bytes size mismatch");
  }

  int num_queries = static_cast<int>(queries.dimensions()[0]);
  if (queries.dimensions()[1] != 3) {
    return xla::ffi::Error::InvalidArgument("queries dim must be 3");
  }

  cudaError_t per_err = SetPeriodicBox(stream, periodic, box_length);
  if (per_err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(per_err));
  }

  const Leaf3* leaf_data = reinterpret_cast<const Leaf3*>(leaf_bytes.untyped_data());
  const Node3* nodes = reinterpret_cast<const Node3*>(node_bytes.untyped_data());
  const float* queries_ptr = queries.typed_data();
  float* out_ptr = out->typed_data();

  int threads = 64;
  int blocks = (num_queries + threads - 1) / threads;
  barnes_hut_gravity_kernel_3d<<<blocks, threads, 0, stream>>>(
      leaf_data, nodes, queries_ptr, num_queries, beta, out_ptr);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(err));
  }

  return xla::ffi::Error::Success();
}

inline cudaError_t SetSofteningCutoff(cudaStream_t stream, float softening,
                                      float cutoff) {
  cudaError_t err = cudaMemcpyToSymbolAsync(
      g_softening_scale, &softening, sizeof(float), 0, cudaMemcpyHostToDevice,
      stream);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaMemcpyToSymbolAsync(g_cutoff_scale, &cutoff, sizeof(float), 0,
                                cudaMemcpyHostToDevice, stream);
  return err;
}

inline cudaError_t SetPeriodicBox(cudaStream_t stream, int periodic,
                                  float box_length) {
  cudaError_t err = cudaMemcpyToSymbolAsync(
      g_periodic_min_image, &periodic, sizeof(int), 0, cudaMemcpyHostToDevice,
      stream);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaMemcpyToSymbolAsync(
      g_periodic_box_length, &box_length, sizeof(float), 0,
      cudaMemcpyHostToDevice, stream);
  return err;
}

xla::ffi::Error BarnesHutForce(cudaStream_t stream,
                               U8R1 leaf_bytes,
                               U8R1 node_bytes,
                               F32R2 queries,
                               F32R2Out out,
                               float beta,
                               int periodic,
                               float box_length) {
  if (leaf_bytes.size_bytes() % sizeof(Leaf2) != 0) {
    return xla::ffi::Error::InvalidArgument("leaf_bytes size mismatch");
  }
  if (node_bytes.size_bytes() % sizeof(Node2) != 0) {
    return xla::ffi::Error::InvalidArgument("node_bytes size mismatch");
  }

  int num_queries = static_cast<int>(queries.dimensions()[0]);
  if (queries.dimensions()[1] != 2) {
    return xla::ffi::Error::InvalidArgument("queries dim must be 2");
  }
  auto out_dims = out->dimensions();
  if (out_dims[0] != num_queries || out_dims[1] != 2) {
    return xla::ffi::Error::InvalidArgument("out shape mismatch");
  }

  cudaError_t per_err = SetPeriodicBox(stream, periodic, box_length);
  if (per_err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(per_err));
  }

  const Leaf2* leaf_data = reinterpret_cast<const Leaf2*>(leaf_bytes.untyped_data());
  const Node2* nodes = reinterpret_cast<const Node2*>(node_bytes.untyped_data());
  const float* queries_ptr = queries.typed_data();
  float* out_ptr = out->typed_data();

  int threads = 256;
  int blocks = (num_queries + threads - 1) / threads;
  barnes_hut_force_kernel<<<blocks, threads, 0, stream>>>(
      leaf_data, nodes, queries_ptr, num_queries, beta, out_ptr);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(err));
  }

  return xla::ffi::Error::Success();
}

xla::ffi::Error SoftenedBarnesHutForce(cudaStream_t stream,
                                       U8R1 leaf_bytes,
                                       U8R1 node_bytes,
                                       F32R2 queries,
                                       F32R2Out out,
                                       float beta,
                                       float softening_scale,
                                       float cutoff_scale,
                                       int periodic,
                                       float box_length) {
  if (leaf_bytes.size_bytes() % sizeof(Leaf2) != 0) {
    return xla::ffi::Error::InvalidArgument("leaf_bytes size mismatch");
  }
  if (node_bytes.size_bytes() % sizeof(Node2) != 0) {
    return xla::ffi::Error::InvalidArgument("node_bytes size mismatch");
  }

  int num_queries = static_cast<int>(queries.dimensions()[0]);
  if (queries.dimensions()[1] != 2) {
    return xla::ffi::Error::InvalidArgument("queries dim must be 2");
  }
  auto out_dims = out->dimensions();
  if (out_dims[0] != num_queries || out_dims[1] != 2) {
    return xla::ffi::Error::InvalidArgument("out shape mismatch");
  }

  cudaError_t per_err = SetPeriodicBox(stream, periodic, box_length);
  if (per_err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(per_err));
  }

  cudaError_t set_err = SetSofteningCutoff(stream, softening_scale, cutoff_scale);
  if (set_err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(set_err));
  }

  const Leaf2* leaf_data =
      reinterpret_cast<const Leaf2*>(leaf_bytes.untyped_data());
  const Node2* nodes = reinterpret_cast<const Node2*>(node_bytes.untyped_data());
  const float* queries_ptr = queries.typed_data();
  float* out_ptr = out->typed_data();

  int threads = 256;
  int blocks = (num_queries + threads - 1) / threads;
  barnes_hut_force_soft_kernel<<<blocks, threads, 0, stream>>>(
      leaf_data, nodes, queries_ptr, num_queries, beta, out_ptr);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(err));
  }

  return xla::ffi::Error::Success();
}

xla::ffi::Error SoftenedBarnesHutForce3D(cudaStream_t stream,
                                         U8R1 leaf_bytes,
                                         U8R1 node_bytes,
                                         F32R2 queries,
                                         F32R2Out out,
                                         float beta,
                                         float softening_scale,
                                         float cutoff_scale,
                                         int periodic,
                                         float box_length) {
  if (leaf_bytes.size_bytes() % sizeof(Leaf3) != 0) {
    return xla::ffi::Error::InvalidArgument("leaf_bytes size mismatch");
  }
  if (node_bytes.size_bytes() % sizeof(Node3) != 0) {
    return xla::ffi::Error::InvalidArgument("node_bytes size mismatch");
  }

  int num_queries = static_cast<int>(queries.dimensions()[0]);
  if (queries.dimensions()[1] != 3) {
    return xla::ffi::Error::InvalidArgument("queries dim must be 3");
  }
  auto out_dims = out->dimensions();
  if (out_dims[0] != num_queries || out_dims[1] != 3) {
    return xla::ffi::Error::InvalidArgument("out shape mismatch");
  }

  cudaError_t per_err = SetPeriodicBox(stream, periodic, box_length);
  if (per_err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(per_err));
  }

  cudaError_t set_err = SetSofteningCutoff(stream, softening_scale, cutoff_scale);
  if (set_err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(set_err));
  }

  const Leaf3* leaf_data =
      reinterpret_cast<const Leaf3*>(leaf_bytes.untyped_data());
  const Node3* nodes = reinterpret_cast<const Node3*>(node_bytes.untyped_data());
  const float* queries_ptr = queries.typed_data();
  float* out_ptr = out->typed_data();

  int threads = 256;
  int blocks = (num_queries + threads - 1) / threads;
  barnes_hut_force_soft_kernel_3d<<<blocks, threads, 0, stream>>>(
      leaf_data, nodes, queries_ptr, num_queries, beta, out_ptr);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(err));
  }

  return xla::ffi::Error::Success();
}

xla::ffi::Error BarnesHutForce3D(cudaStream_t stream,
                                 U8R1 leaf_bytes,
                                 U8R1 node_bytes,
                                 F32R2 queries,
                                 F32R2Out out,
                                 float beta,
                                 int periodic,
                                 float box_length) {
  if (leaf_bytes.size_bytes() % sizeof(Leaf3) != 0) {
    return xla::ffi::Error::InvalidArgument("leaf_bytes size mismatch");
  }
  if (node_bytes.size_bytes() % sizeof(Node3) != 0) {
    return xla::ffi::Error::InvalidArgument("node_bytes size mismatch");
  }

  int num_queries = static_cast<int>(queries.dimensions()[0]);
  if (queries.dimensions()[1] != 3) {
    return xla::ffi::Error::InvalidArgument("queries dim must be 3");
  }
  auto out_dims = out->dimensions();
  if (out_dims[0] != num_queries || out_dims[1] != 3) {
    return xla::ffi::Error::InvalidArgument("out shape mismatch");
  }

  cudaError_t per_err = SetPeriodicBox(stream, periodic, box_length);
  if (per_err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(per_err));
  }

  const Leaf3* leaf_data = reinterpret_cast<const Leaf3*>(leaf_bytes.untyped_data());
  const Node3* nodes = reinterpret_cast<const Node3*>(node_bytes.untyped_data());
  const float* queries_ptr = queries.typed_data();
  float* out_ptr = out->typed_data();

  int threads = 256;
  int blocks = (num_queries + threads - 1) / threads;
  barnes_hut_force_kernel_3d<<<blocks, threads, 0, stream>>>(
      leaf_data, nodes, queries_ptr, num_queries, beta, out_ptr);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(err));
  }

  return xla::ffi::Error::Success();
}

xla::ffi::Error StochasticBarnesHutGravity(cudaStream_t stream,
                                           U8R1 leaf_bytes,
                                           U8R1 node_bytes,
                                           U8R1 contrib_bytes,
                                           F32R2 queries,
                                           F32R1Out out,
                                           int samples_per_subdomain,
                                           int seed,
                                           int periodic,
                                           float box_length) {
  if (leaf_bytes.size_bytes() % sizeof(Leaf2) != 0) {
    return xla::ffi::Error::InvalidArgument("leaf_bytes size mismatch");
  }
  if (node_bytes.size_bytes() % sizeof(Node2) != 0) {
    return xla::ffi::Error::InvalidArgument("node_bytes size mismatch");
  }
  if (contrib_bytes.size_bytes() % sizeof(Leaf2) != 0) {
    return xla::ffi::Error::InvalidArgument("contrib_bytes size mismatch");
  }

  int num_queries = static_cast<int>(queries.dimensions()[0]);
  if (queries.dimensions()[1] != 2) {
    return xla::ffi::Error::InvalidArgument("queries dim must be 2");
  }

  cudaError_t per_err = SetPeriodicBox(stream, periodic, box_length);
  if (per_err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(per_err));
  }

  int num_points = static_cast<int>(leaf_bytes.size_bytes() / sizeof(Leaf2));

  const Leaf2* leaf_data = reinterpret_cast<const Leaf2*>(leaf_bytes.untyped_data());
  const Node2* nodes = reinterpret_cast<const Node2*>(node_bytes.untyped_data());
  const Leaf2* contrib = reinterpret_cast<const Leaf2*>(contrib_bytes.untyped_data());
  const float* queries_ptr = queries.typed_data();
  float* out_ptr = out->typed_data();

  int threads = 256;
  int blocks = (num_queries + threads - 1) / threads;
  stochastic_bh_gravity_kernel_2d<<<blocks, threads, 0, stream>>>(
      leaf_data, num_points, nodes, contrib, queries_ptr, num_queries,
      samples_per_subdomain, seed, out_ptr);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(err));
  }

  return xla::ffi::Error::Success();
}

xla::ffi::Error StochasticBarnesHutGravity3D(cudaStream_t stream,
                                             U8R1 leaf_bytes,
                                             U8R1 node_bytes,
                                             U8R1 contrib_bytes,
                                             F32R2 queries,
                                             F32R1Out out,
                                             int samples_per_subdomain,
                                             int seed,
                                             int periodic,
                                             float box_length) {
  if (leaf_bytes.size_bytes() % sizeof(Leaf3) != 0) {
    return xla::ffi::Error::InvalidArgument("leaf_bytes size mismatch");
  }
  if (node_bytes.size_bytes() % sizeof(Node3) != 0) {
    return xla::ffi::Error::InvalidArgument("node_bytes size mismatch");
  }
  if (contrib_bytes.size_bytes() % sizeof(Leaf3) != 0) {
    return xla::ffi::Error::InvalidArgument("contrib_bytes size mismatch");
  }

  int num_queries = static_cast<int>(queries.dimensions()[0]);
  if (queries.dimensions()[1] != 3) {
    return xla::ffi::Error::InvalidArgument("queries dim must be 3");
  }

  cudaError_t per_err = SetPeriodicBox(stream, periodic, box_length);
  if (per_err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(per_err));
  }

  int num_points = static_cast<int>(leaf_bytes.size_bytes() / sizeof(Leaf3));

  const Leaf3* leaf_data = reinterpret_cast<const Leaf3*>(leaf_bytes.untyped_data());
  const Node3* nodes = reinterpret_cast<const Node3*>(node_bytes.untyped_data());
  const Leaf3* contrib = reinterpret_cast<const Leaf3*>(contrib_bytes.untyped_data());
  const float* queries_ptr = queries.typed_data();
  float* out_ptr = out->typed_data();

  int threads = 256;
  int blocks = (num_queries + threads - 1) / threads;
  stochastic_bh_gravity_kernel_3d<<<blocks, threads, 0, stream>>>(
      leaf_data, num_points, nodes, contrib, queries_ptr, num_queries,
      samples_per_subdomain, seed, out_ptr);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(err));
  }

  return xla::ffi::Error::Success();
}

xla::ffi::Error StochasticBarnesHutForce(cudaStream_t stream,
                                         U8R1 leaf_bytes,
                                         U8R1 node_bytes,
                                         U8R1 contrib_bytes,
                                         F32R2 queries,
                                         F32R2Out out,
                                         int samples_per_subdomain,
                                         int seed,
                                         int periodic,
                                         float box_length) {
  if (leaf_bytes.size_bytes() % sizeof(Leaf2) != 0) {
    return xla::ffi::Error::InvalidArgument("leaf_bytes size mismatch");
  }
  if (node_bytes.size_bytes() % sizeof(Node2) != 0) {
    return xla::ffi::Error::InvalidArgument("node_bytes size mismatch");
  }
  if (contrib_bytes.size_bytes() % sizeof(Leaf2) != 0) {
    return xla::ffi::Error::InvalidArgument("contrib_bytes size mismatch");
  }

  int num_queries = static_cast<int>(queries.dimensions()[0]);
  if (queries.dimensions()[1] != 2) {
    return xla::ffi::Error::InvalidArgument("queries dim must be 2");
  }

  cudaError_t per_err = SetPeriodicBox(stream, periodic, box_length);
  if (per_err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(per_err));
  }

  int num_points = static_cast<int>(leaf_bytes.size_bytes() / sizeof(Leaf2));

  const Leaf2* leaf_data = reinterpret_cast<const Leaf2*>(leaf_bytes.untyped_data());
  const Node2* nodes = reinterpret_cast<const Node2*>(node_bytes.untyped_data());
  const Leaf2* contrib = reinterpret_cast<const Leaf2*>(contrib_bytes.untyped_data());
  const float* queries_ptr = queries.typed_data();
  float* out_ptr = out->typed_data();

  int threads = 256;
  int blocks = (num_queries + threads - 1) / threads;
  stochastic_bh_force_kernel_2d<<<blocks, threads, 0, stream>>>(
      leaf_data, num_points, nodes, contrib, queries_ptr, num_queries,
      samples_per_subdomain, seed, out_ptr);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(err));
  }

  return xla::ffi::Error::Success();
}

xla::ffi::Error StochasticBarnesHutForce3D(cudaStream_t stream,
                                           U8R1 leaf_bytes,
                                           U8R1 node_bytes,
                                           U8R1 contrib_bytes,
                                           F32R2 queries,
                                           F32R2Out out,
                                           int samples_per_subdomain,
                                           int seed,
                                           int periodic,
                                           float box_length) {
  if (leaf_bytes.size_bytes() % sizeof(Leaf3) != 0) {
    return xla::ffi::Error::InvalidArgument("leaf_bytes size mismatch");
  }
  if (node_bytes.size_bytes() % sizeof(Node3) != 0) {
    return xla::ffi::Error::InvalidArgument("node_bytes size mismatch");
  }
  if (contrib_bytes.size_bytes() % sizeof(Leaf3) != 0) {
    return xla::ffi::Error::InvalidArgument("contrib_bytes size mismatch");
  }

  int num_queries = static_cast<int>(queries.dimensions()[0]);
  if (queries.dimensions()[1] != 3) {
    return xla::ffi::Error::InvalidArgument("queries dim must be 3");
  }

  cudaError_t per_err = SetPeriodicBox(stream, periodic, box_length);
  if (per_err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(per_err));
  }

  int num_points = static_cast<int>(leaf_bytes.size_bytes() / sizeof(Leaf3));

  const Leaf3* leaf_data = reinterpret_cast<const Leaf3*>(leaf_bytes.untyped_data());
  const Node3* nodes = reinterpret_cast<const Node3*>(node_bytes.untyped_data());
  const Leaf3* contrib = reinterpret_cast<const Leaf3*>(contrib_bytes.untyped_data());
  const float* queries_ptr = queries.typed_data();
  float* out_ptr = out->typed_data();

  int threads = 256;
  int blocks = (num_queries + threads - 1) / threads;
  stochastic_bh_force_kernel_3d<<<blocks, threads, 0, stream>>>(
      leaf_data, num_points, nodes, contrib, queries_ptr, num_queries,
      samples_per_subdomain, seed, out_ptr);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(err));
  }

  return xla::ffi::Error::Success();
}

xla::ffi::Error SoftenedStochasticBarnesHutForce(cudaStream_t stream,
                                                 U8R1 leaf_bytes,
                                                 U8R1 node_bytes,
                                                 U8R1 contrib_bytes,
                                                 F32R2 queries,
                                                 F32R2Out out,
                                                 int samples_per_subdomain,
                                                 int seed,
                                                 float softening_scale,
                                                 float cutoff_scale,
                                                 int periodic,
                                                 float box_length) {
  if (leaf_bytes.size_bytes() % sizeof(Leaf2) != 0) {
    return xla::ffi::Error::InvalidArgument("leaf_bytes size mismatch");
  }
  if (node_bytes.size_bytes() % sizeof(Node2) != 0) {
    return xla::ffi::Error::InvalidArgument("node_bytes size mismatch");
  }
  if (contrib_bytes.size_bytes() % sizeof(Leaf2) != 0) {
    return xla::ffi::Error::InvalidArgument("contrib_bytes size mismatch");
  }

  int num_queries = static_cast<int>(queries.dimensions()[0]);
  if (queries.dimensions()[1] != 2) {
    return xla::ffi::Error::InvalidArgument("queries dim must be 2");
  }
  auto out_dims = out->dimensions();
  if (out_dims[0] != num_queries || out_dims[1] != 2) {
    return xla::ffi::Error::InvalidArgument("out shape mismatch");
  }

  cudaError_t per_err = SetPeriodicBox(stream, periodic, box_length);
  if (per_err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(per_err));
  }

  cudaError_t set_err = SetSofteningCutoff(stream, softening_scale, cutoff_scale);
  if (set_err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(set_err));
  }

  int num_points = static_cast<int>(leaf_bytes.size_bytes() / sizeof(Leaf2));
  const Leaf2* leaf_data =
      reinterpret_cast<const Leaf2*>(leaf_bytes.untyped_data());
  const Node2* nodes = reinterpret_cast<const Node2*>(node_bytes.untyped_data());
  const Leaf2* contrib =
      reinterpret_cast<const Leaf2*>(contrib_bytes.untyped_data());
  const float* queries_ptr = queries.typed_data();
  float* out_ptr = out->typed_data();

  int threads = 256;
  int blocks = (num_queries + threads - 1) / threads;
  stochastic_bh_force_soft_kernel_2d<<<blocks, threads, 0, stream>>>(
      leaf_data, num_points, nodes, contrib, queries_ptr, num_queries,
      samples_per_subdomain, seed, out_ptr);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(err));
  }

  return xla::ffi::Error::Success();
}

xla::ffi::Error SoftenedStochasticBarnesHutForce3D(cudaStream_t stream,
                                                   U8R1 leaf_bytes,
                                                   U8R1 node_bytes,
                                                   U8R1 contrib_bytes,
                                                   F32R2 queries,
                                                   F32R2Out out,
                                                   int samples_per_subdomain,
                                                   int seed,
                                                   float softening_scale,
                                                   float cutoff_scale,
                                                   int periodic,
                                                   float box_length) {
  if (leaf_bytes.size_bytes() % sizeof(Leaf3) != 0) {
    return xla::ffi::Error::InvalidArgument("leaf_bytes size mismatch");
  }
  if (node_bytes.size_bytes() % sizeof(Node3) != 0) {
    return xla::ffi::Error::InvalidArgument("node_bytes size mismatch");
  }
  if (contrib_bytes.size_bytes() % sizeof(Leaf3) != 0) {
    return xla::ffi::Error::InvalidArgument("contrib_bytes size mismatch");
  }

  int num_queries = static_cast<int>(queries.dimensions()[0]);
  if (queries.dimensions()[1] != 3) {
    return xla::ffi::Error::InvalidArgument("queries dim must be 3");
  }
  auto out_dims = out->dimensions();
  if (out_dims[0] != num_queries || out_dims[1] != 3) {
    return xla::ffi::Error::InvalidArgument("out shape mismatch");
  }

  cudaError_t per_err = SetPeriodicBox(stream, periodic, box_length);
  if (per_err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(per_err));
  }

  cudaError_t set_err = SetSofteningCutoff(stream, softening_scale, cutoff_scale);
  if (set_err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(set_err));
  }

  int num_points = static_cast<int>(leaf_bytes.size_bytes() / sizeof(Leaf3));
  const Leaf3* leaf_data =
      reinterpret_cast<const Leaf3*>(leaf_bytes.untyped_data());
  const Node3* nodes = reinterpret_cast<const Node3*>(node_bytes.untyped_data());
  const Leaf3* contrib =
      reinterpret_cast<const Leaf3*>(contrib_bytes.untyped_data());
  const float* queries_ptr = queries.typed_data();
  float* out_ptr = out->typed_data();

  int threads = 256;
  int blocks = (num_queries + threads - 1) / threads;
  stochastic_bh_force_soft_kernel_3d<<<blocks, threads, 0, stream>>>(
      leaf_data, num_points, nodes, contrib, queries_ptr, num_queries,
      samples_per_subdomain, seed, out_ptr);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(err));
  }

  return xla::ffi::Error::Success();
}

xla::ffi::Error BarnesHutForceOctree8(cudaStream_t stream,
                                      U8R1 leaf_bytes,
                                      U8R1 node_bytes,
                                      F32R2 queries,
                                      F32R2Out out,
                                      float beta,
                                      int periodic,
                                      float box_length) {
  if (leaf_bytes.size_bytes() % sizeof(Leaf3) != 0) {
    return xla::ffi::Error::InvalidArgument("leaf_bytes size mismatch");
  }
  if (node_bytes.size_bytes() % sizeof(OctreeNode8) != 0) {
    return xla::ffi::Error::InvalidArgument("node_bytes size mismatch");
  }
  int num_queries = static_cast<int>(queries.dimensions()[0]);
  if (queries.dimensions()[1] != 3) {
    return xla::ffi::Error::InvalidArgument("queries dim must be 3");
  }
  auto out_dims = out->dimensions();
  if (out_dims[0] != num_queries || out_dims[1] != 3) {
    return xla::ffi::Error::InvalidArgument("out shape mismatch");
  }

  cudaError_t per_err = SetPeriodicBox(stream, periodic, box_length);
  if (per_err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(per_err));
  }

  const Leaf3* leaf_data =
      reinterpret_cast<const Leaf3*>(leaf_bytes.untyped_data());
  const OctreeNode8* nodes =
      reinterpret_cast<const OctreeNode8*>(node_bytes.untyped_data());
  const float* queries_ptr = queries.typed_data();
  float* out_ptr = out->typed_data();

  int threads = 256;
  int blocks = (num_queries + threads - 1) / threads;
  barnes_hut_force_octree8_kernel<<<blocks, threads, 0, stream>>>(
      leaf_data, nodes, queries_ptr, num_queries, beta, out_ptr);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(err));
  }
  return xla::ffi::Error::Success();
}

xla::ffi::Error SoftenedBarnesHutForceOctree8(cudaStream_t stream,
                                              U8R1 leaf_bytes,
                                              U8R1 node_bytes,
                                              F32R2 queries,
                                              F32R2Out out,
                                              float beta,
                                              float softening_scale,
                                              float cutoff_scale,
                                              int periodic,
                                              float box_length) {
  if (leaf_bytes.size_bytes() % sizeof(Leaf3) != 0) {
    return xla::ffi::Error::InvalidArgument("leaf_bytes size mismatch");
  }
  if (node_bytes.size_bytes() % sizeof(OctreeNode8) != 0) {
    return xla::ffi::Error::InvalidArgument("node_bytes size mismatch");
  }
  int num_queries = static_cast<int>(queries.dimensions()[0]);
  if (queries.dimensions()[1] != 3) {
    return xla::ffi::Error::InvalidArgument("queries dim must be 3");
  }
  auto out_dims = out->dimensions();
  if (out_dims[0] != num_queries || out_dims[1] != 3) {
    return xla::ffi::Error::InvalidArgument("out shape mismatch");
  }

  cudaError_t per_err = SetPeriodicBox(stream, periodic, box_length);
  if (per_err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(per_err));
  }

  cudaError_t set_err = SetSofteningCutoff(stream, softening_scale, cutoff_scale);
  if (set_err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(set_err));
  }

  const Leaf3* leaf_data =
      reinterpret_cast<const Leaf3*>(leaf_bytes.untyped_data());
  const OctreeNode8* nodes =
      reinterpret_cast<const OctreeNode8*>(node_bytes.untyped_data());
  const float* queries_ptr = queries.typed_data();
  float* out_ptr = out->typed_data();

  int threads = 256;
  int blocks = (num_queries + threads - 1) / threads;
  barnes_hut_force_octree8_soft_kernel<<<blocks, threads, 0, stream>>>(
      leaf_data, nodes, queries_ptr, num_queries, beta, out_ptr);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(err));
  }
  return xla::ffi::Error::Success();
}

xla::ffi::Error SoftenedBarnesHutForceOctree8Vjp(cudaStream_t stream,
                                                 U8R1 leaf_bytes,
                                                 U8R1 node_bytes,
                                                 F32R2 queries,
                                                 F32R2 cotangent,
                                                 F32R2Out out,
                                                 float beta,
                                                 float softening_scale,
                                                 float cutoff_scale,
                                                 int periodic,
                                                 float box_length) {
  if (leaf_bytes.size_bytes() % sizeof(Leaf3) != 0) {
    return xla::ffi::Error::InvalidArgument("leaf_bytes size mismatch");
  }
  if (node_bytes.size_bytes() % sizeof(OctreeNode8) != 0) {
    return xla::ffi::Error::InvalidArgument("node_bytes size mismatch");
  }
  int num_queries = static_cast<int>(queries.dimensions()[0]);
  if (queries.dimensions()[1] != 3) {
    return xla::ffi::Error::InvalidArgument("queries dim must be 3");
  }
  if (cotangent.dimensions()[0] != num_queries || cotangent.dimensions()[1] != 3) {
    return xla::ffi::Error::InvalidArgument("cotangent shape mismatch");
  }
  auto out_dims = out->dimensions();
  if (out_dims[0] != num_queries || out_dims[1] != 3) {
    return xla::ffi::Error::InvalidArgument("out shape mismatch");
  }

  cudaError_t per_err = SetPeriodicBox(stream, periodic, box_length);
  if (per_err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(per_err));
  }

  cudaError_t set_err = SetSofteningCutoff(stream, softening_scale, cutoff_scale);
  if (set_err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(set_err));
  }

  const Leaf3* leaf_data =
      reinterpret_cast<const Leaf3*>(leaf_bytes.untyped_data());
  const OctreeNode8* nodes =
      reinterpret_cast<const OctreeNode8*>(node_bytes.untyped_data());
  const float* queries_ptr = queries.typed_data();
  const float* cotangent_ptr = cotangent.typed_data();
  float* out_ptr = out->typed_data();

  int threads = 256;
  int blocks = (num_queries + threads - 1) / threads;
  barnes_hut_force_octree8_soft_vjp_kernel<<<blocks, threads, 0, stream>>>(
      leaf_data, nodes, queries_ptr, cotangent_ptr, num_queries, beta, out_ptr);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(err));
  }
  return xla::ffi::Error::Success();
}

xla::ffi::Error StochasticBarnesHutForceOctree8(cudaStream_t stream,
                                                U8R1 leaf_bytes,
                                                U8R1 node_bytes,
                                                F32R2 queries,
                                                F32R2Out out,
                                                int samples_per_subdomain,
                                                int seed,
                                                int periodic,
                                                float box_length) {
  if (leaf_bytes.size_bytes() % sizeof(Leaf3) != 0) {
    return xla::ffi::Error::InvalidArgument("leaf_bytes size mismatch");
  }
  if (node_bytes.size_bytes() % sizeof(OctreeNode8) != 0) {
    return xla::ffi::Error::InvalidArgument("node_bytes size mismatch");
  }
  int num_queries = static_cast<int>(queries.dimensions()[0]);
  if (queries.dimensions()[1] != 3) {
    return xla::ffi::Error::InvalidArgument("queries dim must be 3");
  }
  auto out_dims = out->dimensions();
  if (out_dims[0] != num_queries || out_dims[1] != 3) {
    return xla::ffi::Error::InvalidArgument("out shape mismatch");
  }

  cudaError_t per_err = SetPeriodicBox(stream, periodic, box_length);
  if (per_err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(per_err));
  }

  const Leaf3* leaf_data =
      reinterpret_cast<const Leaf3*>(leaf_bytes.untyped_data());
  const OctreeNode8* nodes =
      reinterpret_cast<const OctreeNode8*>(node_bytes.untyped_data());
  const float* queries_ptr = queries.typed_data();
  float* out_ptr = out->typed_data();

  int threads = 256;
  int blocks = (num_queries + threads - 1) / threads;
  stochastic_bh_force_octree8_kernel<<<blocks, threads, 0, stream>>>(
      leaf_data, nodes, queries_ptr, num_queries, samples_per_subdomain, seed,
      out_ptr, false);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(err));
  }
  return xla::ffi::Error::Success();
}

xla::ffi::Error SoftenedStochasticBarnesHutForceOctree8(
    cudaStream_t stream,
    U8R1 leaf_bytes,
    U8R1 node_bytes,
    F32R2 queries,
    F32R2Out out,
    int samples_per_subdomain,
    int seed,
    float softening_scale,
    float cutoff_scale,
    int periodic,
    float box_length) {
  if (leaf_bytes.size_bytes() % sizeof(Leaf3) != 0) {
    return xla::ffi::Error::InvalidArgument("leaf_bytes size mismatch");
  }
  if (node_bytes.size_bytes() % sizeof(OctreeNode8) != 0) {
    return xla::ffi::Error::InvalidArgument("node_bytes size mismatch");
  }
  int num_queries = static_cast<int>(queries.dimensions()[0]);
  if (queries.dimensions()[1] != 3) {
    return xla::ffi::Error::InvalidArgument("queries dim must be 3");
  }
  auto out_dims = out->dimensions();
  if (out_dims[0] != num_queries || out_dims[1] != 3) {
    return xla::ffi::Error::InvalidArgument("out shape mismatch");
  }

  cudaError_t per_err = SetPeriodicBox(stream, periodic, box_length);
  if (per_err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(per_err));
  }

  cudaError_t set_err = SetSofteningCutoff(stream, softening_scale, cutoff_scale);
  if (set_err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(set_err));
  }

  const Leaf3* leaf_data =
      reinterpret_cast<const Leaf3*>(leaf_bytes.untyped_data());
  const OctreeNode8* nodes =
      reinterpret_cast<const OctreeNode8*>(node_bytes.untyped_data());
  const float* queries_ptr = queries.typed_data();
  float* out_ptr = out->typed_data();

  int threads = 256;
  int blocks = (num_queries + threads - 1) / threads;
  stochastic_bh_force_octree8_kernel<<<blocks, threads, 0, stream>>>(
      leaf_data, nodes, queries_ptr, num_queries, samples_per_subdomain, seed,
      out_ptr, true);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return xla::ffi::Error::Internal(cudaGetErrorString(err));
  }
  return xla::ffi::Error::Success();
}

}  // namespace sbh_ffi

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sbh_bruteforce_gravity_ffi, sbh_ffi::BruteForceGravity,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<sbh_ffi::F32R2>()
        .Arg<sbh_ffi::F32R1>()
        .Arg<sbh_ffi::F32R2>()
        .Ret<sbh_ffi::F32R1>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sbh_build_octree_level3_buffers_ffi, sbh_ffi::BuildOctreeLevel3Buffers,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<sbh_ffi::F32R2>()
        .Arg<sbh_ffi::F32R2>()
        .Arg<sbh_ffi::F32R1>()
        .Arg<sbh_ffi::F32R1>()
        .Arg<sbh_ffi::F32R1>()
        .Ret<sbh_ffi::U8R1>()
        .Ret<sbh_ffi::U8R1>()
        .Ret<sbh_ffi::U8R1>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sbh_build_octree8_buffers_ffi, sbh_ffi::BuildOctree8Buffers,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<sbh_ffi::F32R2>()
        .Arg<sbh_ffi::F32R2>()
        .Arg<sbh_ffi::F32R1>()
        .Arg<sbh_ffi::F32R1>()
        .Arg<sbh_ffi::F32R1>()
        .Ret<sbh_ffi::U8R1>()
        .Ret<sbh_ffi::U8R1>()
        .Ret<sbh_ffi::U8R1>()
        .Attr<int>("max_depth"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sbh_barnes_hut_gravity_ffi, sbh_ffi::BarnesHutGravity,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::F32R2>()
        .Ret<sbh_ffi::F32R1>()
        .Attr<float>("beta")
        .Attr<int>("periodic")
        .Attr<float>("box_length"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sbh_barnes_hut_force_ffi, sbh_ffi::BarnesHutForce,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::F32R2>()
        .Ret<sbh_ffi::F32R2>()
        .Attr<float>("beta")
        .Attr<int>("periodic")
        .Attr<float>("box_length"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sbh_barnes_hut_gravity_ffi_3d, sbh_ffi::BarnesHutGravity3D,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::F32R2>()
        .Ret<sbh_ffi::F32R1>()
        .Attr<float>("beta")
        .Attr<int>("periodic")
        .Attr<float>("box_length"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sbh_barnes_hut_force_ffi_3d, sbh_ffi::BarnesHutForce3D,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::F32R2>()
        .Ret<sbh_ffi::F32R2>()
        .Attr<float>("beta")
        .Attr<int>("periodic")
        .Attr<float>("box_length"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sbh_barnes_hut_force_octree8_ffi, sbh_ffi::BarnesHutForceOctree8,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::F32R2>()
        .Ret<sbh_ffi::F32R2>()
        .Attr<float>("beta")
        .Attr<int>("periodic")
        .Attr<float>("box_length"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sbh_barnes_hut_force_octree8_soft_ffi,
    sbh_ffi::SoftenedBarnesHutForceOctree8,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::F32R2>()
        .Ret<sbh_ffi::F32R2>()
        .Attr<float>("beta")
        .Attr<float>("softening_scale")
        .Attr<float>("cutoff_scale")
        .Attr<int>("periodic")
        .Attr<float>("box_length"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sbh_barnes_hut_force_octree8_soft_vjp_ffi,
    sbh_ffi::SoftenedBarnesHutForceOctree8Vjp,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::F32R2>()
        .Arg<sbh_ffi::F32R2>()
        .Ret<sbh_ffi::F32R2>()
        .Attr<float>("beta")
        .Attr<float>("softening_scale")
        .Attr<float>("cutoff_scale")
        .Attr<int>("periodic")
        .Attr<float>("box_length"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sbh_barnes_hut_force_soft_ffi, sbh_ffi::SoftenedBarnesHutForce,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::F32R2>()
        .Ret<sbh_ffi::F32R2>()
        .Attr<float>("beta")
        .Attr<float>("softening_scale")
        .Attr<float>("cutoff_scale")
        .Attr<int>("periodic")
        .Attr<float>("box_length"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sbh_barnes_hut_force_soft_ffi_3d, sbh_ffi::SoftenedBarnesHutForce3D,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::F32R2>()
        .Ret<sbh_ffi::F32R2>()
        .Attr<float>("beta")
        .Attr<float>("softening_scale")
        .Attr<float>("cutoff_scale")
        .Attr<int>("periodic")
        .Attr<float>("box_length"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sbh_stochastic_bh_gravity_ffi, sbh_ffi::StochasticBarnesHutGravity,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::F32R2>()
        .Ret<sbh_ffi::F32R1>()
        .Attr<int>("samples_per_subdomain")
        .Attr<int>("seed")
        .Attr<int>("periodic")
        .Attr<float>("box_length"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sbh_stochastic_bh_gravity_ffi_3d, sbh_ffi::StochasticBarnesHutGravity3D,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::F32R2>()
        .Ret<sbh_ffi::F32R1>()
        .Attr<int>("samples_per_subdomain")
        .Attr<int>("seed")
        .Attr<int>("periodic")
        .Attr<float>("box_length"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sbh_stochastic_bh_force_ffi, sbh_ffi::StochasticBarnesHutForce,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::F32R2>()
        .Ret<sbh_ffi::F32R2>()
        .Attr<int>("samples_per_subdomain")
        .Attr<int>("seed")
        .Attr<int>("periodic")
        .Attr<float>("box_length"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sbh_stochastic_bh_force_ffi_3d, sbh_ffi::StochasticBarnesHutForce3D,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::F32R2>()
        .Ret<sbh_ffi::F32R2>()
        .Attr<int>("samples_per_subdomain")
        .Attr<int>("seed")
        .Attr<int>("periodic")
        .Attr<float>("box_length"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sbh_stochastic_bh_force_octree8_ffi,
    sbh_ffi::StochasticBarnesHutForceOctree8,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::F32R2>()
        .Ret<sbh_ffi::F32R2>()
        .Attr<int>("samples_per_subdomain")
        .Attr<int>("seed")
        .Attr<int>("periodic")
        .Attr<float>("box_length"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sbh_stochastic_bh_force_octree8_soft_ffi,
    sbh_ffi::SoftenedStochasticBarnesHutForceOctree8,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::F32R2>()
        .Ret<sbh_ffi::F32R2>()
        .Attr<int>("samples_per_subdomain")
        .Attr<int>("seed")
        .Attr<float>("softening_scale")
        .Attr<float>("cutoff_scale")
        .Attr<int>("periodic")
        .Attr<float>("box_length"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sbh_stochastic_bh_force_soft_ffi, sbh_ffi::SoftenedStochasticBarnesHutForce,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::F32R2>()
        .Ret<sbh_ffi::F32R2>()
        .Attr<int>("samples_per_subdomain")
        .Attr<int>("seed")
        .Attr<float>("softening_scale")
        .Attr<float>("cutoff_scale")
        .Attr<int>("periodic")
        .Attr<float>("box_length"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sbh_stochastic_bh_force_soft_ffi_3d,
    sbh_ffi::SoftenedStochasticBarnesHutForce3D,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::U8R1>()
        .Arg<sbh_ffi::F32R2>()
        .Ret<sbh_ffi::F32R2>()
        .Attr<int>("samples_per_subdomain")
        .Attr<int>("seed")
        .Attr<float>("softening_scale")
        .Attr<float>("cutoff_scale")
        .Attr<int>("periodic")
        .Attr<float>("box_length"));

extern "C" size_t sbh_leaf3_size() {
  return sizeof(sbh_ffi::Leaf3);
}

extern "C" size_t sbh_node3_size() {
  return sizeof(sbh_ffi::Node3);
}

extern "C" size_t sbh_octree_level3_max_nodes() {
  return static_cast<size_t>(sbh_ffi::kOctreeMaxNodes);
}

extern "C" size_t sbh_octree8_node_size() {
  return sizeof(sbh_ffi::OctreeNode8);
}
