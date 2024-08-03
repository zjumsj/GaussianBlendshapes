#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <c10/cuda/CUDAGuard.h> // support multiple GPUs

#include <algorithm>
#include <stdexcept>

#include <stdint.h>
#include <cstdio>


__global__ void kernel_getIdAndWeight(
    const float * __restrict__ inputs,
    const int * __restrict__ grid,
    int * __restrict__ outputs_id,
    float * __restrict__ outputs_w,
    const int32_t B, const uint32_t D, const uint32_t H, const uint32_t W, bool align_corners
){
    const int32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= B) return;
    float pos[3] = {
        inputs[id * 3 + 0],
        inputs[id * 3 + 1],
        inputs[id * 3 + 2]
    };
    // map coord from [-1,1] -> [0,1] -> [0,Ni]
    if (align_corners){
        pos[0] = (pos[0] * 0.5f + 0.5f) * float(W - 1);
        pos[1] = (pos[1] * 0.5f + 0.5f) * float(H - 1);
        pos[2] = (pos[2] * 0.5f + 0.5f) * float(D - 1);
    }else{
        pos[0] = (pos[0] * 0.5f + 0.5f) * float(W);
        pos[1] = (pos[1] * 0.5f + 0.5f) * float(H);
        pos[2] = (pos[2] * 0.5f + 0.5f) * float(D);
    }
    int u_min[3];
    int u_max[3];
    float k[3];
#pragma unroll
    for(int p = 0; p < 3; p++){
        if(align_corners){
            u_min[p] = int(floor(pos[p]));
            k[p] = pos[p] - float(u_min[p]);
        }else{
            u_min[p] = int(floor(pos[p] - 0.5f));
            k[p] = pos[p] - float(u_min[p]) - 0.5f;
        }
        u_max[p] = u_min[p] + 1;
        if(u_min[p] < 0) u_min[p] = 0;
        if(u_max[p] < 0) u_max[p] = 0;
    }
    if(u_min[0] >= W) u_min[0] = W - 1;
    if(u_max[0] >= W) u_max[0] = W - 1;
    if(u_min[1] >= H) u_min[1] = H - 1;
    if(u_max[1] >= H) u_max[1] = H - 1;
    if(u_min[2] >= D) u_min[2] = D - 1;
    if(u_max[2] >= D) u_max[2] = D - 1;
    int v000 = grid[(u_min[2] * H + u_min[1]) * W + u_min[0]];
    int v001 = grid[(u_min[2] * H + u_min[1]) * W + u_max[0]];
    int v010 = grid[(u_min[2] * H + u_max[1]) * W + u_min[0]];
    int v011 = grid[(u_min[2] * H + u_max[1]) * W + u_max[0]];
    int v100 = grid[(u_max[2] * H + u_min[1]) * W + u_min[0]];
    int v101 = grid[(u_max[2] * H + u_min[1]) * W + u_max[0]];
    int v110 = grid[(u_max[2] * H + u_max[1]) * W + u_min[0]];
    int v111 = grid[(u_max[2] * H + u_max[1]) * W + u_max[0]];
    float k000 = (1-k[2])*(1-k[1])*(1-k[0]);
    float k001 = (1-k[2])*(1-k[1])*k[0];
    float k010 = (1-k[2])*k[1]*(1-k[0]);
    float k011 = (1-k[2])*k[1]*k[0];
    float k100 = k[2]*(1-k[1])*(1-k[0]);
    float k101 = k[2]*(1-k[1])*k[0];
    float k110 = k[2]*k[1]*(1-k[0]);
    float k111 = k[2]*k[1]*k[0];

    outputs_id[id * 8 + 0] = v000;
    outputs_id[id * 8 + 1] = v001;
    outputs_id[id * 8 + 2] = v010;
    outputs_id[id * 8 + 3] = v011;
    outputs_id[id * 8 + 4] = v100;
    outputs_id[id * 8 + 5] = v101;
    outputs_id[id * 8 + 6] = v110;
    outputs_id[id * 8 + 7] = v111;

    outputs_w[id * 8 + 0] = k000;
    outputs_w[id * 8 + 1] = k001;
    outputs_w[id * 8 + 2] = k010;
    outputs_w[id * 8 + 3] = k011;
    outputs_w[id * 8 + 4] = k100;
    outputs_w[id * 8 + 5] = k101;
    outputs_w[id * 8 + 6] = k110;
    outputs_w[id * 8 + 7] = k111;
}

///////////////////

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")

template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

std::vector<torch::Tensor> getIdAndWeight(
    const torch::Tensor& in_points, // Nx3
    const torch::Tensor& grid, // int32, DHW
    bool align_corners
){
    CHECK_CUDA(in_points); CHECK_CUDA(grid); //CHECK_CUDA(aabb);
    CHECK_CONTIGUOUS(in_points); CHECK_CONTIGUOUS(grid); //CHECK_CONTIGUOUS(aabb);
    CHECK_IS_FLOATING(in_points); //CHECK_IS_FLOATING(aabb);

    int64_t B = in_points.size(0);
    int64_t D = grid.size(0);
    int64_t H = grid.size(1);
    int64_t W = grid.size(2);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(in_points));
    auto device = in_points.device();

    at::TensorOptions opt_i(grid.dtype()); opt_i = opt_i.device(device); opt_i = opt_i.requires_grad(false);
    at::TensorOptions opt(in_points.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);

    torch::Tensor outputs_id = torch::empty({B, 8}, opt_i);
    torch::Tensor outputs_w = torch::empty({B, 8}, opt);

    const uint32_t THREADS = 256;
    const uint32_t BLOCKS = div_round_up((uint32_t)B, THREADS);

    kernel_getIdAndWeight<<<BLOCKS,THREADS>>>(
        in_points.data_ptr<float>(),
        grid.data_ptr<int>(),
        outputs_id.data_ptr<int>(),
        outputs_w.data_ptr<float>(),
        B, D, H, W, align_corners
    );
    return {outputs_id, outputs_w};
}
