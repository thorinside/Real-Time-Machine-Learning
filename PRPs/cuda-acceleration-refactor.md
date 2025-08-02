# PRP: CUDA Acceleration for Learning Feature Trees

## Feature Overview
Refactor the Learning Feature Trees (LF-Trees) implementation to use CUDA acceleration while maintaining CPU compatibility. The goal is to achieve the 200-20,000x speedup potential mentioned in the codebase while ensuring the code can still run on CPU-only systems.

## Executive Summary
This PRP outlines a comprehensive refactoring to add CUDA GPU acceleration to the existing Learning Feature Trees implementation. The approach uses a hybrid CPU/GPU architecture with runtime device detection, unified interfaces, and optimized memory management patterns. Based on research, similar implementations have achieved 10-140x speedups for tree-based ML algorithms.

## Research Findings and Context

### Performance Bottlenecks Identified
1. **Sequential Tree Growth** - Current implementation grows trees one at a time (wolf.h:190-301, test_wolf.cpp:79-101)
2. **Nested Loop Structures** - Multiple nested loops over 60,000 images × 784 sensors (wolf.h:236-290)
3. **Memory Access Patterns** - Non-coalesced access to image pixels (wolf.h:257, wolf.h:367)
4. **Feature Computation** - Repeated computation for each image (wolf.h:438, wolf.h:469)
5. **Tree Traversal** - Sequential block finding during evaluation (wolf.h:642-669)

### CUDA Best Practices for Tree-Based ML
- **RAPIDS cuML** achieves 10-50x speedup: https://github.com/rapidsai/cuml
- **Breadth-first tree construction** reduces warp divergence
- **Structure of Arrays (SoA)** layout for coalesced memory access
- **Unified Memory** with prefetching for dynamic tree growth

### Memory Management Patterns
- **Pinned Memory**: 2-3x bandwidth improvement for CPU-GPU transfers
- **Memory Pooling**: 1.24x training speedup with proper allocation strategies
- **Coalesced Access**: 5-55x speedup for tree algorithms
- **CUDA Streams**: 50% reduction in execution time with overlapped operations

## Implementation Blueprint

### Architecture Design

```cpp
// Core abstraction layer
class ComputeDevice {
public:
    enum Type { CPU, CUDA };
    virtual Type getType() const = 0;
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual void memcpy(void* dst, const void* src, size_t bytes, bool async = false) = 0;
    virtual void synchronize() = 0;
};

// Template-based algorithm implementation
template<typename DeviceType>
class LF_tree_impl : public cLF_tree {
    DeviceType device;
    
    // Device-specific memory for trees
    struct DeviceMemory {
        float* centroids;      // SoA layout for coalescing
        float* weights;
        int* active_sensors;
        int* image_indices;
        // Tree structure in SoA format
        float* feature_thresholds;
        int* left_children;
        int* right_children;
    };
};

// Runtime selection
std::unique_ptr<cLF_tree> create_LF_tree() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount > 0 && std::getenv("WOLF_USE_CUDA") != "0") {
        return std::make_unique<LF_tree_impl<CUDADevice>>();
    } else {
        return std::make_unique<LF_tree_impl<CPUDevice>>();
    }
}
```

### Parallel Tree Growing Strategy

```cpp
// CUDA kernel for parallel tree growth
__global__ void growTreesKernel(
    TreeData* trees,
    ImageData* images,
    int num_trees,
    int num_images,
    int num_sensors
) {
    int tree_id = blockIdx.x;
    if (tree_id >= num_trees) return;
    
    // Each block handles one tree
    // Use shared memory for frequently accessed data
    __shared__ float shared_centroids[784];
    __shared__ int shared_active[784];
    
    // Collaborative loading by threads in block
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    for (int i = tid; i < num_sensors; i += stride) {
        shared_centroids[i] = trees[tree_id].centroids[i];
        shared_active[i] = trees[tree_id].active_sensors[i];
    }
    __syncthreads();
    
    // Tree growth logic with warp-level primitives
    // ...
}

// Host code for launching parallel tree growth
void growAllTreesCUDA(int num_trees) {
    dim3 blocks(num_trees);
    dim3 threads(256);  // Optimal for most GPUs
    
    // Use CUDA streams for overlapped execution
    cudaStream_t* streams = new cudaStream_t[4];
    for (int i = 0; i < 4; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Launch trees in batches across streams
    int trees_per_stream = (num_trees + 3) / 4;
    for (int s = 0; s < 4; s++) {
        int start = s * trees_per_stream;
        int count = min(trees_per_stream, num_trees - start);
        
        growTreesKernel<<<count, 256, 0, streams[s]>>>(
            d_trees + start,
            d_images,
            count,
            60000,
            784
        );
    }
    
    // Synchronize all streams
    for (int i = 0; i < 4; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;
}
```

### Memory Optimization Patterns

```cpp
// Optimized data structures for GPU
struct SampleBlockGPU {
    // Structure of Arrays for coalesced access
    struct {
        float* values;      // [num_blocks][num_sensors]
        int* block_ids;     // [num_blocks]
    } centroids;
    
    struct {
        float* values;      // [num_blocks][num_sensors]  
        int* active_mask;   // [num_blocks][num_sensors/32] bit mask
    } weights;
    
    struct {
        int* indices;       // [total_images]
        int* block_start;   // [num_blocks + 1] CSR format
    } images;
};

// Unified memory allocation with hints
template<typename T>
T* allocateUnified(size_t count) {
    T* ptr;
    size_t bytes = count * sizeof(T);
    
    cudaMallocManaged(&ptr, bytes);
    
    // Performance hints
    cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, 0);
    cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, 0);
    
    return ptr;
}

// Pinned memory for fast transfers
class PinnedMemoryPool {
    std::vector<void*> free_blocks;
    std::mutex mutex;
    
public:
    void* allocate(size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex);
        
        if (!free_blocks.empty() && bytes <= BLOCK_SIZE) {
            void* ptr = free_blocks.back();
            free_blocks.pop_back();
            return ptr;
        }
        
        void* ptr;
        cudaMallocHost(&ptr, bytes);
        return ptr;
    }
    
    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex);
        free_blocks.push_back(ptr);
    }
};
```

### Feature Computation Optimization

```cpp
// Optimized feature computation kernel
__global__ void computeFeaturesCoalesced(
    float* image_data,      // [num_images][num_sensors] 
    float* features_out,    // [num_images][num_features]
    FeatureSpec* specs,     // Feature definitions
    int num_images,
    int num_sensors,
    int num_features
) {
    // Grid-stride loop for arbitrary batch sizes
    int img_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Shared memory for feature specs
    extern __shared__ FeatureSpec shared_specs[];
    
    // Collaborative loading of feature specs
    int tid = threadIdx.x;
    for (int i = tid; i < num_features; i += blockDim.x) {
        shared_specs[i] = specs[i];
    }
    __syncthreads();
    
    // Process images
    for (int img = img_idx; img < num_images; img += stride) {
        // Compute all features for this image
        for (int f = 0; f < num_features; f++) {
            float sum = 0.0f;
            
            // Coalesced access pattern
            #pragma unroll 4
            for (int s = 0; s < shared_specs[f].num_sensors; s++) {
                int sensor_idx = shared_specs[f].sensor_indices[s];
                float factor = shared_specs[f].factors[s];
                sum += image_data[img * num_sensors + sensor_idx] * factor;
            }
            
            features_out[img * num_features + f] = sum;
        }
    }
}
```

### CPU Fallback Implementation

```cpp
// CPU implementation with same interface
class CPUDevice : public ComputeDevice {
public:
    Type getType() const override { return CPU; }
    
    void* allocate(size_t bytes) override {
        return std::aligned_alloc(64, bytes);  // Cache-line aligned
    }
    
    void deallocate(void* ptr) override {
        std::free(ptr);
    }
    
    void memcpy(void* dst, const void* src, size_t bytes, bool async) override {
        std::memcpy(dst, src, bytes);
    }
    
    void synchronize() override {
        // No-op for CPU
    }
};

// CPU parallel implementation using OpenMP
void growTreesCPU(cLF_tree** trees, int num_trees) {
    #pragma omp parallel for
    for (int i = 0; i < num_trees; i++) {
        trees[i]->loadSBs(60000);
        trees[i]->growTree();
    }
}
```

## Implementation Tasks

1. **Create Device Abstraction Layer** (wolf_device.h)
   - Define ComputeDevice interface
   - Implement CPUDevice and CUDADevice classes
   - Add runtime device detection

2. **Refactor Data Structures** (wolf_gpu_types.h)
   - Convert Sample Blocks to Structure of Arrays format
   - Implement unified memory allocators
   - Create pinned memory pool

3. **Implement CUDA Kernels** (wolf_kernels.cu)
   - Parallel tree growth kernel
   - Feature computation kernel
   - Tree evaluation kernel
   - Convolution kernel

4. **Update Core Algorithm** (wolf.h)
   - Template cLF_tree class on device type
   - Add device memory management
   - Implement hybrid CPU/GPU logic

5. **Optimize Memory Transfers** (wolf_memory.cu)
   - Implement CUDA streams
   - Add prefetching logic
   - Optimize data layout

6. **Update Main Application** (test_wolf.cpp)
   - Add command-line GPU selection
   - Implement parallel tree launching
   - Add performance timing

7. **Create Build System** (CMakeLists.txt)
   - Add CUDA detection
   - Conditional compilation
   - Link appropriate libraries

## Validation Gates

```bash
# 1. Build validation - Both CPU and CUDA
mkdir build && cd build

# CPU-only build
cmake .. -DENABLE_CUDA=OFF
make -j8
./RTML MNIST 784 10 10 15.0 2.0

# CUDA build
cmake .. -DENABLE_CUDA=ON
make -j8

# 2. Runtime device detection
export WOLF_USE_CUDA=0
./RTML MNIST 784 10 10 15.0 2.0  # Should use CPU

export WOLF_USE_CUDA=1
./RTML MNIST 784 10 10 15.0 2.0  # Should use GPU

# 3. Correctness validation
# Compare outputs between CPU and GPU versions
./RTML MNIST 784 200 10 15.0 2.0 > cpu_output.txt
WOLF_USE_CUDA=1 ./RTML MNIST 784 200 10 15.0 2.0 > gpu_output.txt
diff cpu_output.txt gpu_output.txt

# 4. Performance benchmarks
# Measure speedup
time ./RTML MNIST 784 200 10 15.0 2.0
time WOLF_USE_CUDA=1 ./RTML MNIST 784 200 10 15.0 2.0

# 5. Memory leak detection
cuda-memcheck ./RTML MNIST 784 10 10 15.0 2.0

# 6. Unit tests
./test_wolf_gpu --gtest_filter=*GPU*
./test_wolf_cpu --gtest_filter=*CPU*

# 7. Profiling
nsys profile --stats=true ./RTML MNIST 784 200 10 15.0 2.0
ncu --set full ./RTML MNIST 784 10 10 15.0 2.0
```

## Error Handling Strategy

1. **Device Initialization Failures**
   - Graceful fallback to CPU
   - Clear error messages
   - Runtime warnings

2. **Memory Allocation Failures**
   - Try smaller batch sizes
   - Use unified memory fallback
   - Report memory requirements

3. **Kernel Launch Failures**
   - Validate grid/block dimensions
   - Check shared memory limits
   - Provide detailed CUDA errors

## Performance Expectations

Based on research and similar implementations:
- **Single Tree Growth**: 50-100x speedup
- **Parallel Tree Growth**: 200-500x speedup for 200 trees
- **Feature Computation**: 20-50x speedup
- **Tree Evaluation**: 10-30x speedup
- **Overall Training**: 100-200x speedup
- **Memory Usage**: 1.5-2x of CPU version

## Additional Resources

- NVIDIA CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- RAPIDS cuML Random Forest: https://github.com/rapidsai/cuml/tree/main/cpp/src/randomforest
- ArrayFire ML Examples: https://github.com/arrayfire/arrayfire/tree/master/examples/machine_learning
- CUDA Samples: https://github.com/NVIDIA/cuda-samples

## Risk Mitigation

1. **Compatibility Risk**: Extensive CPU fallback testing
2. **Performance Risk**: Multiple optimization strategies
3. **Memory Risk**: Adaptive batch sizing
4. **Maintenance Risk**: Clean abstraction layers

## Success Metrics

1. Maintains 96% MNIST accuracy
2. Achieves minimum 50x speedup on GPU
3. Zero performance regression on CPU
4. Passes all existing tests
5. Clean compilation on both paths

---

**Confidence Score: 9/10**

This PRP provides comprehensive context for successful one-pass implementation with self-validation capabilities. The research depth, concrete examples, and clear implementation path significantly increase the likelihood of success.