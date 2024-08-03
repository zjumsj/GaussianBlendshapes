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

#include "sh_rot.cuh"

// Gaussian splatting's SH definition is different from
// wiki https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
// Enable this flag to do transformation
#define NEG_SH_SIGN

__device__ void grad_inv_of_mat3x3(const float* src, const float* grad_out, float* grad_in) {

    float det = src[0] * (src[4] * src[8] - src[7] * src[5]) -
        src[1] * (src[3] * src[8] - src[5] * src[6]) +
        src[2] * (src[3] * src[7] - src[4] * src[6]);
    float invdet = 1.f / det;
    //
    float grad_invdet = 0.f;
    for (int i = 0; i < 9; i++)
        grad_in[i] = 0.f;
    //
    grad_in[4] += grad_out[0] * src[8] * invdet; grad_in[8] += grad_out[0] * src[4] * invdet; grad_in[7] -= grad_out[0] * src[5] * invdet; grad_in[5] -= grad_out[0] * src[7] * invdet;
    grad_invdet += grad_out[0] * (src[4] * src[8] - src[7] * src[5]);
    grad_in[2] += grad_out[1] * src[7] * invdet; grad_in[7] += grad_out[1] * src[2] * invdet; grad_in[1] -= grad_out[1] * src[8] * invdet; grad_in[8] -= grad_out[1] * src[1] * invdet;
    grad_invdet += grad_out[1] * (src[2] * src[7] - src[1] * src[8]);
    grad_in[1] += grad_out[2] * src[5] * invdet; grad_in[5] += grad_out[2] * src[1] * invdet; grad_in[2] -= grad_out[2] * src[4] * invdet; grad_in[4] -= grad_out[2] * src[2] * invdet;
    grad_invdet += grad_out[2] * (src[1] * src[5] - src[2] * src[4]);
    //
    grad_in[5] += grad_out[3] * src[6] * invdet; grad_in[6] += grad_out[3] * src[5] * invdet; grad_in[3] -= grad_out[3] * src[8] * invdet; grad_in[8] -= grad_out[3] * src[3] * invdet;
    grad_invdet += grad_out[3] * (src[5] * src[6] - src[3] * src[8]);
    grad_in[0] += grad_out[4] * src[8] * invdet; grad_in[8] += grad_out[4] * src[0] * invdet; grad_in[2] -= grad_out[4] * src[6] * invdet; grad_in[6] -= grad_out[4] * src[2] * invdet;
    grad_invdet += grad_out[4] * (src[0] * src[8] - src[2] * src[6]);
    grad_in[3] += grad_out[5] * src[2] * invdet; grad_in[2] += grad_out[5] * src[3] * invdet; grad_in[0] -= grad_out[5] * src[5] * invdet; grad_in[5] -= grad_out[5] * src[0] * invdet;
    grad_invdet += grad_out[5] * (src[3] * src[2] - src[0] * src[5]);
    //
    grad_in[3] += grad_out[6] * src[7] * invdet; grad_in[7] += grad_out[6] * src[3] * invdet; grad_in[6] -= grad_out[6] * src[4] * invdet; grad_in[4] -= grad_out[6] * src[6] * invdet;
    grad_invdet += grad_out[6] * (src[3] * src[7] - src[6] * src[4]);
    grad_in[6] += grad_out[7] * src[1] * invdet; grad_in[1] += grad_out[7] * src[6] * invdet; grad_in[0] -= grad_out[7] * src[7] * invdet; grad_in[7] -= grad_out[7] * src[0] * invdet;
    grad_invdet += grad_out[7] * (src[6] * src[1] - src[0] * src[7]);
    grad_in[0] += grad_out[8] * src[4] * invdet; grad_in[4] += grad_out[8] * src[0] * invdet; grad_in[3] -= grad_out[8] * src[1] * invdet; grad_in[1] -= grad_out[8] * src[3] * invdet;
    grad_invdet += grad_out[8] * (src[0] * src[4] - src[3] * src[1]);

    //
    grad_invdet = -1 / (det * det) * grad_invdet; // grad_det
    grad_in[0] += grad_invdet * (src[4] * src[8] - src[7] * src[5]);
    grad_in[1] -= grad_invdet * (src[3] * src[8] - src[5] * src[6]);
    grad_in[2] += grad_invdet * (src[3] * src[7] - src[4] * src[6]);
    grad_in[4] += grad_invdet * src[0] * src[8]; grad_in[8] += grad_invdet * src[0] * src[4]; grad_in[7] -= grad_invdet * src[0] * src[5]; grad_in[5] -= grad_invdet * src[0] * src[7];
    grad_in[3] -= grad_invdet * src[1] * src[8]; grad_in[8] -= grad_invdet * src[1] * src[3]; grad_in[5] += grad_invdet * src[1] * src[6]; grad_in[6] += grad_invdet * src[1] * src[5];
    grad_in[3] += grad_invdet * src[2] * src[7]; grad_in[7] += grad_invdet * src[2] * src[3]; grad_in[4] -= grad_invdet * src[2] * src[6]; grad_in[6] -= grad_invdet * src[2] * src[4];
}

__device__ void inv_of_mat3x3(const float* src, float* tar) {

    float det = src[0] * (src[4] * src[8] - src[7] * src[5]) -
        src[1] * (src[3] * src[8] - src[5] * src[6]) +
        src[2] * (src[3] * src[7] - src[4] * src[6]);
    float invdet = 1.f / det;
    tar[0] = (src[4] * src[8] - src[7] * src[5]) * invdet;
    tar[1] = (src[2] * src[7] - src[1] * src[8]) * invdet;
    tar[2] = (src[1] * src[5] - src[2] * src[4]) * invdet;

    tar[3] = (src[5] * src[6] - src[3] * src[8]) * invdet;
    tar[4] = (src[0] * src[8] - src[2] * src[6]) * invdet;
    tar[5] = (src[3] * src[2] - src[0] * src[5]) * invdet;

    tar[6] = (src[3] * src[7] - src[6] * src[4]) * invdet;
    tar[7] = (src[6] * src[1] - src[0] * src[7]) * invdet;
    tar[8] = (src[0] * src[4] - src[3] * src[1]) * invdet;
}

__device__ void quaternion_to_matrix(float* mat, const float* quat) {
	float r = quat[0];
	float i = quat[1];
	float j = quat[2];
	float k = quat[3];
	float two_s = 2.f / (r * r + i * i + j * j + k * k);
	mat[0] = 1.f - two_s * (j * j + k * k);
	mat[1] = two_s * (i * j - k * r);
	mat[2] = two_s * (i * k + j * r);
	mat[3] = two_s * (i * j + k * r);
	mat[4] = 1.f - two_s * (i * i + k * k);
	mat[5] = two_s * (j * k - i * r);
	mat[6] = two_s * (i * k - j * r);
	mat[7] = two_s * (j * k + i * r);
	mat[8] = 1.f - two_s * (i * i + j * j);
}

// input Bx3x3
__global__ void __launch_bounds__(128) InvMat3x3Kernel_Backward(
    int64_t n_elements,
    const float* __restrict__ in_mat,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_in
){
    __shared__ float s_mat[128 * 9];
    int64_t thread_idx = blockIdx.x * 128 + threadIdx.x;
    float loc_in_mat[9];
    float loc_grad_in_mat[9];
    float loc_grad_out_mat[9];
    //// load first data
#pragma unroll
    for(int i = 0; i < 9; i++){
        int loc_offset = i * 128 + threadIdx.x;
        int64_t i_elem = blockIdx.x * (9*128) + loc_offset;
        if(i_elem < n_elements * 9){
            s_mat[loc_offset] = in_mat[i_elem];
        }
    }
    __syncthreads();
    if(thread_idx < n_elements){
#pragma unroll
        for(int j = 0; j < 9; j++){
            loc_in_mat[j] = s_mat[threadIdx.x * 9 + j];
        }
    }
    //// load second data
    __syncthreads();
#pragma unroll
    for(int i = 0; i < 9; i++){
        int loc_offset = i * 128 + threadIdx.x;
        int64_t i_elem = blockIdx.x * (9*128) + loc_offset;
        if(i_elem < n_elements * 9){
            s_mat[loc_offset] = grad_out[i_elem];
        }
    }
    __syncthreads();
    if(thread_idx < n_elements){
#pragma unroll
        for(int j = 0; j < 9; j++){
            loc_grad_out_mat[j] = s_mat[threadIdx.x * 9 + j];
        }
        grad_inv_of_mat3x3(loc_in_mat, loc_grad_out_mat, loc_grad_in_mat);
    }
    //// write back, first to shared memory, then to global memory
    __syncthreads();
    if(thread_idx < n_elements){
#pragma unroll
        for(int j = 0; j < 9; j++){
            s_mat[threadIdx.x * 9 + j] = loc_grad_in_mat[j];
        }
    }
    __syncthreads();
#pragma unroll
    for(int i = 0; i < 9; i++){
        int loc_offset = i * 128 + threadIdx.x;
        int64_t i_elem = blockIdx.x * (9*128) + loc_offset;
        if(i_elem < n_elements * 9){
            grad_in[i_elem] = s_mat[loc_offset];
        }
    }
}

// input Bx3x3
__global__ void __launch_bounds__(128) InvMat3x3Kernel(
    int64_t n_elements,
    const float* __restrict__ in_mat,
    float * out_mat
){
    __shared__ float s_mat[128 * 9];
    int64_t thread_idx = blockIdx.x * 128 + threadIdx.x;
    float loc_in_mat[9];
    float loc_out_mat[9];

#pragma unroll
    for(int i = 0; i < 9; i++){
        int loc_offset = i * 128 + threadIdx.x;
        int64_t i_elem = blockIdx.x * (9*128) + loc_offset;
        if(i_elem < n_elements * 9){
            s_mat[loc_offset] = in_mat[i_elem];
        }
    }
    __syncthreads();

    if(thread_idx < n_elements){
#pragma unroll
        for(int j = 0; j < 9; j++){
            loc_in_mat[j] = s_mat[threadIdx.x * 9 + j];
        }
        inv_of_mat3x3(loc_in_mat, loc_out_mat);
    }
    //// write back, first to shared memory, then to global memory
    __syncthreads();
    if(thread_idx < n_elements){
#pragma unroll
        for(int j = 0; j < 9; j++){
            s_mat[threadIdx.x * 9 + j] = loc_out_mat[j];
        }
    }
    __syncthreads();
#pragma unroll
    for(int i = 0; i < 9; i++){
        int loc_offset = i * 128 + threadIdx.x;
        int64_t i_elem = blockIdx.x * (9*128) + loc_offset;
        if(i_elem < n_elements * 9){
            out_mat[i_elem] = s_mat[loc_offset];
        }
    }
}

// threads = 32 * 3
// shared memory for sh = 32 * 3 * 16
// shared memory for matrix = 32 * 9
__global__ void __launch_bounds__(32*3) CompositeBasisRotSHKernel_Backward(
	int64_t n_elements, //int p_offset, int p_n, int p_total,
	const float* __restrict__ faceNR,
	const float* __restrict__ sh,
	const float* __restrict__ grad_shs_out, // {P}xLx3
	float* __restrict__ grad_faceNR,
	float* __restrict__ grad_shs_in // {P}xLx3
){
    __shared__ float s_shs[32 * 3 * 16];
    //__shared__ float s_mat[32 * 9];
    __shared__ float s_mat[32 * 3 * 9];
#pragma unroll
	for (int i = 0; i < 16; i++) {
		int loc_offset = i * 96 + threadIdx.x;
		int64_t i_elem = blockIdx.x * (96 * 16) + loc_offset;
		if (i_elem < n_elements * 48) {
			s_shs[loc_offset] = sh[i_elem];
		}
	}
#pragma unroll
	for (int i = 0; i < 3; i++) {
		int loc_offset = i * 96 + threadIdx.x;
		int64_t i_elem = blockIdx.x * (96 * 3) + loc_offset;
		if (i_elem < n_elements * 9) {
			s_mat[loc_offset] = faceNR[i_elem];
		}
	}
	__syncthreads();
	// read sh to local
	int64_t thread_idx = blockIdx.x * 96 + threadIdx.x;
	int64_t i_elem = thread_idx / 3;
	int li_elem = threadIdx.x / 3;
	int channel = threadIdx.x % 3;
    float loc_sh1[16]; // input sh & output grad_of_sh
    float loc_mat[9];
    if(i_elem < n_elements){
#pragma unroll
		for (int i = 0; i < 16; i++) {
			loc_sh1[i] = s_shs[li_elem * 48 + i * 3 + channel];
		}
#ifdef NEG_SH_SIGN
#pragma unroll
        for (int i = 0; i < 16; i++) {
            if (i % 2) loc_sh1[i] = -loc_sh1[i];
        }
#endif
    }
    __syncthreads();
    // read grad of sh to shared memory
#pragma unroll
	for (int i = 0; i < 16; i++) {
		int loc_offset = i * 96 + threadIdx.x;
		int64_t i_elem = blockIdx.x * (96 * 16) + loc_offset;
		if (i_elem < n_elements * 48) {
			s_shs[loc_offset] = grad_shs_out[i_elem];
		}
	}
	__syncthreads();
	// read shared memory
	if (i_elem < n_elements){
	    float loc_sh[16]; // input gradient
#pragma unroll
		for (int i = 0; i < 16; i++) {
			loc_sh[i] = s_shs[li_elem * 48 + i * 3 + channel];
		}
#ifdef NEG_SH_SIGN
#pragma unroll
        for (int i = 0; i < 16; i++) {
            if (i % 2) loc_sh[i] = -loc_sh[i];
        }
#endif
#pragma unroll
		for (int i = 0; i < 9; i++) {
			loc_mat[i] = s_mat[li_elem * 9 + i];
		}

		float sh1[3][3];
	    float sh2[5][5];
	    float sh3[7][7];
		Construct_SH_Rotation_Matrix(loc_mat, sh1, sh2, sh3);
        // after
        Construct_SH_Rotation_Matrix_Backward(
            loc_sh, loc_sh1,
            sh1, sh2, sh3, loc_mat
        );
		//
        loc_sh1[0] = loc_sh[0];

		loc_sh1[1] = sh1[0][0] * loc_sh[1] + sh1[1][0] * loc_sh[2] + sh1[2][0] * loc_sh[3];
		loc_sh1[2] = sh1[0][1] * loc_sh[1] + sh1[1][1] * loc_sh[2] + sh1[2][1] * loc_sh[3];
		loc_sh1[3] = sh1[0][2] * loc_sh[1] + sh1[1][2] * loc_sh[2] + sh1[2][2] * loc_sh[3];

#pragma unroll
		for (int i = 0; i < 5; i++) {
			loc_sh1[4 + i] = 0;
#pragma unroll
			for (int j = 0; j < 5; j++) {
				loc_sh1[4 + i] += sh2[j][i] * loc_sh[4 + j];
			}
		}
#pragma unroll
		for (int i = 0; i < 7; i++) {
			loc_sh1[9 + i] = 0;
#pragma unroll
			for (int j = 0; j < 7; j++) {
				loc_sh1[9 + i] += sh3[j][i] * loc_sh[9 + j];
			}
		}
#ifdef NEG_SH_SIGN
#pragma unroll
        for (int i = 0; i < 16; i++) {
            if (i % 2) loc_sh1[i] = -loc_sh1[i];
        }
#endif
	}
    //// write back, first to shared memory, then to global memory
	__syncthreads();
    if (i_elem < n_elements) {
#pragma unroll
		for (int i = 0; i < 16; i++) {
			s_shs[li_elem * 48 + i * 3 + channel] = loc_sh1[i];
		}
#pragma unroll
		for (int i = 0; i < 9; i++) {
			//s_mat[li_elem * 9 + i] = loc_mat[i];
			s_mat[threadIdx.x * 9 + i] = loc_mat[i];
		}
	}
	__syncthreads();
	// merge rot RGB grad to rot
// 	if(threadIdx.x < 32){
// 	    for(int i = 0; i < 9; i++){
// 	        loc_mat[i] = s_mat[threadIdx.x * 27 + 0 + i] + s_mat[threadIdx.x * 27 + 9 + i] + s_mat[threadIdx.x * 27 + 18 + i];
// 	    }
// 	}
// 	__syncthreads();
// 	if(threadIdx.x < 32){
// 	    for(int i = 0; i < 9; i++){
// 	        s_mat[threadIdx.x * 9 + i] = loc_mat[i];
// 	    }
// 	}
	if(threadIdx.x < 32){
	    for(int i = 0; i < 9; i++){
	        loc_mat[i] = s_mat[threadIdx.x * 27 + 0 + i] + s_mat[threadIdx.x * 27 + 9 + i] + s_mat[threadIdx.x * 27 + 18 + i];
	    }
	    __syncwarp();
	    for(int i = 0; i < 9; i++){
	        s_mat[threadIdx.x * 9 + i] = loc_mat[i];
	    }
	}
	__syncthreads();
#pragma unroll
	for (int i = 0; i < 16; i++) {
		int loc_offset = i * 96 + threadIdx.x;
		int64_t i_elem = blockIdx.x * (96 * 16) + loc_offset;
		if (i_elem < n_elements * 48) {
			grad_shs_in[i_elem] = s_shs[loc_offset];
		}
	}
#pragma unroll
	for (int i = 0; i < 3; i++) {
		int loc_offset = i * 96 + threadIdx.x;
		int64_t i_elem = blockIdx.x * (96 * 3) + loc_offset;
		if (i_elem < n_elements * 9) {
			grad_faceNR[i_elem] = s_mat[loc_offset];
		}
	}
}

// threads = 32 * 3
// shared memory for sh = 32 * 3 * 16
// shared memory for matrix = 32 * 9
__global__ void __launch_bounds__(32*3) CompositeBasisRotSHKernel_BackwardSimple(
	int64_t n_elements, //int p_offset, int p_n, int p_total,
	const float* __restrict__ faceNR,
	const float* __restrict__ grad_shs_out, // {P}xLx3
	float* __restrict__ grad_shs_in // {P}xLx3
){
    __shared__ float s_shs[32 * 3 * 16];
    __shared__ float s_mat[32 * 9];
#pragma unroll
	for (int i = 0; i < 16; i++) {
		int loc_offset = i * 96 + threadIdx.x;
		int64_t i_elem = blockIdx.x * (96 * 16) + loc_offset;
		if (i_elem < n_elements * 48) {
			s_shs[loc_offset] = grad_shs_out[i_elem];
		}
	}
#pragma unroll
	for (int i = 0; i < 3; i++) {
		int loc_offset = i * 96 + threadIdx.x;
		int64_t i_elem = blockIdx.x * (96 * 3) + loc_offset;
		if (i_elem < n_elements * 9) {
			s_mat[loc_offset] = faceNR[i_elem];
		}
	}
	__syncthreads();
	int64_t thread_idx = blockIdx.x * 96 + threadIdx.x;
	int64_t i_elem = thread_idx / 3;
	int li_elem = threadIdx.x / 3;
	int channel = threadIdx.x % 3;
	float loc_sh_out[16];
	if (i_elem < n_elements) {
		float loc_sh[16];
	    float loc_mat[9];
#pragma unroll
		for (int i = 0; i < 16; i++) {
			loc_sh[i] = s_shs[li_elem * 48 + i * 3 + channel];
		}
#ifdef NEG_SH_SIGN
#pragma unroll
        for (int i = 0; i < 16; i++){
            if(i % 2) loc_sh[i] = -loc_sh[i];
        }
#endif

#pragma unroll
		for (int i = 0; i < 9; i++) {
			loc_mat[i] = s_mat[li_elem * 9 + i];
		}

		float sh1[3][3];
	    float sh2[5][5];
	    float sh3[7][7];
		Construct_SH_Rotation_Matrix(loc_mat, sh1, sh2, sh3);

		loc_sh_out[0] = loc_sh[0];

		loc_sh_out[1] = sh1[0][0] * loc_sh[1] + sh1[1][0] * loc_sh[2] + sh1[2][0] * loc_sh[3];
		loc_sh_out[2] = sh1[0][1] * loc_sh[1] + sh1[1][1] * loc_sh[2] + sh1[2][1] * loc_sh[3];
		loc_sh_out[3] = sh1[0][2] * loc_sh[1] + sh1[1][2] * loc_sh[2] + sh1[2][2] * loc_sh[3];

#pragma unroll
		for (int i = 0; i < 5; i++) {
			loc_sh_out[4 + i] = 0;
#pragma unroll
			for (int j = 0; j < 5; j++) {
				loc_sh_out[4 + i] += sh2[j][i] * loc_sh[4 + j];
			}
		}

#pragma unroll
		for (int i = 0; i < 7; i++) {
			loc_sh_out[9 + i] = 0;
#pragma unroll
			for (int j = 0; j < 7; j++) {
				loc_sh_out[9 + i] += sh3[j][i] * loc_sh[9 + j];
			}
		}

#ifdef NEG_SH_SIGN
        for(int i = 0; i < 16; i++){
            if(i % 2) loc_sh_out[i] = -loc_sh_out[i];
        }
#endif
	}
	//// write back, first to shared memory, then to global memory
	__syncthreads();
    if (i_elem < n_elements) {
#pragma unroll
		for (int i = 0; i < 16; i++) {
			s_shs[li_elem * 48 + i * 3 + channel] = loc_sh_out[i];
		}
	}
	__syncthreads();
#pragma unroll
	for (int i = 0; i < 16; i++) {
		int loc_offset = i * 96 + threadIdx.x;
		int64_t i_elem = blockIdx.x * (96 * 16) + loc_offset;
		if (i_elem < n_elements * 48) {
			grad_shs_in[i_elem] = s_shs[loc_offset];
		}
	}
}

// threads = 32 * 3
// shared memory for sh = 32 * 3 * 16
// shared memory for matrix = 32 * 9
__global__ void __launch_bounds__(32*3) CompositeBasisRotSHKernel(
	int64_t n_elements, //int p_offset, int p_n, int p_total,
	const float* __restrict__ faceNR,
	const float* __restrict__ shs_in, // {P}xLx3
	float* __restrict__ shs_out // {P}xLx3
){
    __shared__ float s_shs[32 * 3 * 16];
	__shared__ float s_mat[32 * 9];
#pragma unroll
	for (int i = 0; i < 16; i++) {
		int loc_offset = i * 96 + threadIdx.x;
		int64_t i_elem = blockIdx.x * (96 * 16) + loc_offset;
		if (i_elem < n_elements * 48) {
			s_shs[loc_offset] = shs_in[i_elem];
		}
	}
#pragma unroll
	for (int i = 0; i < 3; i++) {
		int loc_offset = i * 96 + threadIdx.x;
		int64_t i_elem = blockIdx.x * (96 * 3) + loc_offset;
		if (i_elem < n_elements * 9) {
			s_mat[loc_offset] = faceNR[i_elem];
		}
	}
	__syncthreads();
    int64_t thread_idx = blockIdx.x * 96 + threadIdx.x;
	int64_t i_elem = thread_idx / 3;
	int li_elem = threadIdx.x / 3;
	int channel = threadIdx.x % 3;
	float loc_sh_out[16];
	if (i_elem < n_elements) {
		float loc_sh[16];
		float loc_mat[9];
#pragma unroll
		for (int i = 0; i < 16; i++) {
			loc_sh[i] = s_shs[li_elem * 48 + i * 3 + channel];
		}

#ifdef NEG_SH_SIGN
#pragma unroll
        for (int i = 0; i < 16; i++){
            if(i % 2) loc_sh[i] = -loc_sh[i];
        }
#endif

#pragma unroll
		for (int i = 0; i < 9; i++) {
			loc_mat[i] = s_mat[li_elem * 9 + i];
		}

		// Eigen Matrix if col major !!
		//loc_mat[0] = s_mat[li_elem * 9 + 0]; loc_mat[1] = s_mat[li_elem * 9 + 3]; loc_mat[2] = s_mat[li_elem * 9 + 6];
		//loc_mat[3] = s_mat[li_elem * 9 + 1]; loc_mat[4] = s_mat[li_elem * 9 + 4]; loc_mat[5] = s_mat[li_elem * 9 + 7];
		//loc_mat[6] = s_mat[li_elem * 9 + 2]; loc_mat[7] = s_mat[li_elem * 9 + 5]; loc_mat[8] = s_mat[li_elem * 9 + 8];

		float sh1[3][3];
		float sh2[5][5];
		float sh3[7][7];
		Construct_SH_Rotation_Matrix(loc_mat, sh1, sh2, sh3);

		loc_sh_out[0] = loc_sh[0];

		loc_sh_out[1] = sh1[0][0] * loc_sh[1] + sh1[0][1] * loc_sh[2] + sh1[0][2] * loc_sh[3];
		loc_sh_out[2] = sh1[1][0] * loc_sh[1] + sh1[1][1] * loc_sh[2] + sh1[1][2] * loc_sh[3];
		loc_sh_out[3] = sh1[2][0] * loc_sh[1] + sh1[2][1] * loc_sh[2] + sh1[2][2] * loc_sh[3];

#pragma unroll
		for (int i = 0; i < 5; i++) {
			loc_sh_out[4 + i] = 0;
#pragma unroll
			for (int j = 0; j < 5; j++) {
				loc_sh_out[4 + i] += sh2[i][j] * loc_sh[4 + j];
			}
		}

#pragma unroll
		for (int i = 0; i < 7; i++) {
			loc_sh_out[9 + i] = 0;
#pragma unroll
			for (int j = 0; j < 7; j++) {
				loc_sh_out[9 + i] += sh3[i][j] * loc_sh[9 + j];
			}
		}

#ifdef NEG_SH_SIGN
        for(int i = 0; i < 16; i++){
            if(i % 2) loc_sh_out[i] = -loc_sh_out[i];
        }
#endif

// 		if(i_elem == 32){
// 		    for(int j = 0; j < 16; j++){
// 		        printf("%.2f ",loc_sh_out[j]);
// 		    }
// 		    printf("\n");
// 		}

	}
	//// write back, first to shared memory, then to global memory
	__syncthreads();
	if (i_elem < n_elements) {
#pragma unroll
		for (int i = 0; i < 16; i++) {
			s_shs[li_elem * 48 + i * 3 + channel] = loc_sh_out[i];
		}
	}
	__syncthreads();
#pragma unroll
	for (int i = 0; i < 16; i++) {
		int loc_offset = i * 96 + threadIdx.x;
		int64_t i_elem = blockIdx.x * (96 * 16) + loc_offset;
		if (i_elem < n_elements * 48) {
			shs_out[i_elem] = s_shs[loc_offset];
		}
	}
}


__global__ void __launch_bounds__(32 * 3) RotSHBlendShapeKernel_Backward(
    int64_t n_elements, int64_t n_blendshape,
    const float* __restrict__ expr, // N
    const float* __restrict__ rot_q, // Fx4xN
    const int * __restrict__ grid_id, // Px8
    const float * __restrict__ grid_w, // Px8
    const float * __restrict__ grad_shs_out, // PxLx3
    float * __restrict__ grad_shs_in,
    bool add_identity
){
    __shared__ float s_shs[32 * 3 * 16];
#pragma unroll
	for (int i = 0; i < 16; i++) {
		int loc_offset = i * 96 + threadIdx.x;
		int64_t i_elem = blockIdx.x * (96 * 16) + loc_offset;
		if (i_elem < n_elements * 48) {
			s_shs[loc_offset] = grad_shs_out[i_elem];
		}
	}
	__syncthreads();
	int64_t thread_idx = blockIdx.x * 96 + threadIdx.x;
	int64_t i_elem = thread_idx / 3;
	int li_elem = threadIdx.x / 3;
	int channel = threadIdx.x % 3;
	float loc_sh_out[16];
	if(i_elem < n_elements){
        // load sh
	    float loc_sh[16];
#pragma unroll
		for (int i = 0; i < 16; i++) {
			loc_sh[i] = s_shs[li_elem * 48 + i * 3 + channel];
		}
#ifdef NEG_SH_SIGN
#pragma unroll
        for (int i = 0; i < 16; i++){
            if (i % 2) loc_sh[i] = -loc_sh[i];
        }
#endif
	    int loc_grid_id[8];
	    float loc_grid_w[8];
	    // load triangle to fetch
#pragma unroll
	    for(int i = 0; i < 8; i++){
	        loc_grid_id[i] = grid_id[i_elem * 8 + i];
	        loc_grid_w[i] = grid_w[i_elem * 8 + i];
	    }
	    float loc_q[4]; // quaternion
		float loc_mat[9]; // quaternion -> rot matrix
        float sh1[3][3];
        float sh2[5][5];
        float sh3[7][7];
        float sh1_[3][3];
        float sh2_[5][5];
        float sh3_[7][7];
        // init
        InitRot3x3(sh1);
        InitRot5x5(sh2);
        InitRot7x7(sh3);
        for(int i_b = 0; i_b < n_blendshape; i_b++){
            float k_expr = expr[i_b];
            for(int j = 0; j < 8; j++){
                int face_id = loc_grid_id[j];
                loc_q[0] = rot_q[(face_id * 4 + 0) * n_blendshape + i_b]; // Fx4xN
                loc_q[1] = rot_q[(face_id * 4 + 1) * n_blendshape + i_b]; // Fx4xN
                loc_q[2] = rot_q[(face_id * 4 + 2) * n_blendshape + i_b]; // Fx4xN
                loc_q[3] = rot_q[(face_id * 4 + 3) * n_blendshape + i_b]; // Fx4xN
                if(add_identity){ // regard q as residual, we should add quat(1,0,0,0)
                    loc_q[0] += 1;
                }
                quaternion_to_matrix(loc_mat, loc_q);
                Construct_SH_Rotation_Matrix(loc_mat, sh1_, sh2_, sh3_);
                // rot(qi,SH) - SH
                {
                    #pragma unroll
                    for(int i = 0; i < 3; i++){
                        sh1_[i][i] -= 1;
                    }
                    #pragma unroll
                    for(int i = 0; i < 5; i++){
                        sh2_[i][i] -= 1;
                    }
                    #pragma unroll
                    for(int i = 0; i < 7; i++){
                        sh3_[i][i] -= 1;
                    }
                }
                //
                MADRot3x3(sh1,sh1_,loc_grid_w[j] * k_expr);
                MADRot5x5(sh2,sh2_,loc_grid_w[j] * k_expr);
                MADRot7x7(sh3,sh3_,loc_grid_w[j] * k_expr);
            }
        }
        // finally we add mean to transform matrix
        {
            #pragma unroll
            for(int i = 0; i < 3; i++){
                sh1[i][i] += 1;
            }
            #pragma unroll
            for(int i = 0; i < 5; i++){
                sh2[i][i] += 1;
            }
            #pragma unroll
            for(int i = 0; i < 7; i++){
                sh3[i][i] += 1;
            }
        }
        // apply rot matrix to SH
        loc_sh_out[0] = loc_sh[0];

        loc_sh_out[1] = sh1[0][0] * loc_sh[1] + sh1[1][0] * loc_sh[2] + sh1[2][0] * loc_sh[3];
		loc_sh_out[2] = sh1[0][1] * loc_sh[1] + sh1[1][1] * loc_sh[2] + sh1[2][1] * loc_sh[3];
		loc_sh_out[3] = sh1[0][2] * loc_sh[1] + sh1[1][2] * loc_sh[2] + sh1[2][2] * loc_sh[3];

		#pragma unroll
		for (int i = 0; i < 5; i++) {
			loc_sh_out[4 + i] = 0;
#pragma unroll
			for (int j = 0; j < 5; j++) {
				loc_sh_out[4 + i] += sh2[j][i] * loc_sh[4 + j];
			}
		}

#pragma unroll
		for (int i = 0; i < 7; i++) {
			loc_sh_out[9 + i] = 0;
#pragma unroll
			for (int j = 0; j < 7; j++) {
				loc_sh_out[9 + i] += sh3[j][i] * loc_sh[9 + j];
			}
		}
#ifdef NEG_SH_SIGN
#pragma unroll
        for (int i = 0; i < 16; i++) {
            if (i % 2) loc_sh_out[i] = -loc_sh_out[i];
        }
#endif
	}
	//// write back, first to shared memory, then to global memory
	__syncthreads();
	if(i_elem < n_elements){
#pragma unroll
		for (int i = 0; i < 16; i++) {
			s_shs[li_elem * 48 + i * 3 + channel] = loc_sh_out[i];
		}
	}
	__syncthreads();
#pragma unroll
	for (int i = 0; i < 16; i++) {
		int loc_offset = i * 96 + threadIdx.x;
		int64_t i_elem = blockIdx.x * (96 * 16) + loc_offset;
		if (i_elem < n_elements * 48) {
			grad_shs_in[i_elem] = s_shs[loc_offset];
		}
	}
}

__global__ void __launch_bounds__(32 * 3) RotSHBlendShapeKernel(
    int64_t n_elements, int64_t n_blendshape,
    const float* __restrict__ expr, // N
    const float* __restrict__ rot_q, // Fx4xN
    const int * __restrict__ grid_id, // Px8
    const float * __restrict__ grid_w, // Px8
    const float * __restrict__ shs_in, // PxLx3
    float * __restrict__ shs_out,
    bool add_identity
){
    __shared__ float s_shs[32 * 3 * 16];
#pragma unroll
	for (int i = 0; i < 16; i++) {
		int loc_offset = i * 96 + threadIdx.x;
		int64_t i_elem = blockIdx.x * (96 * 16) + loc_offset;
		if (i_elem < n_elements * 48) {
			s_shs[loc_offset] = shs_in[i_elem];
		}
	}
	__syncthreads();
	int64_t thread_idx = blockIdx.x * 96 + threadIdx.x;
	int64_t i_elem = thread_idx / 3;
	int li_elem = threadIdx.x / 3;
	int channel = threadIdx.x % 3;
	float loc_sh_out[16];
	if(i_elem < n_elements){
        // load sh
	    float loc_sh[16];
#pragma unroll
		for (int i = 0; i < 16; i++) {
			loc_sh[i] = s_shs[li_elem * 48 + i * 3 + channel];
		}
#ifdef NEG_SH_SIGN
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            if (i % 2) loc_sh[i] = -loc_sh[i];
        }
#endif
	    int loc_grid_id[8];
	    float loc_grid_w[8];
	    // load triangle to fetch
#pragma unroll
	    for(int i = 0; i < 8; i++){
	        loc_grid_id[i] = grid_id[i_elem * 8 + i];
	        loc_grid_w[i] = grid_w[i_elem * 8 + i];
	    }
	    float loc_q[4]; // quaternion
		float loc_mat[9]; // quaternion -> rot matrix
        float sh1[3][3];
        float sh2[5][5];
        float sh3[7][7];
        float sh1_[3][3];
        float sh2_[5][5];
        float sh3_[7][7];
        // init
        InitRot3x3(sh1);
        InitRot5x5(sh2);
        InitRot7x7(sh3);
        for(int i_b = 0; i_b < n_blendshape; i_b++){
            float k_expr = expr[i_b];
            for(int j = 0; j < 8; j++){
                int face_id = loc_grid_id[j];
                loc_q[0] = rot_q[(face_id * 4 + 0) * n_blendshape + i_b]; // Fx4xN
                loc_q[1] = rot_q[(face_id * 4 + 1) * n_blendshape + i_b]; // Fx4xN
                loc_q[2] = rot_q[(face_id * 4 + 2) * n_blendshape + i_b]; // Fx4xN
                loc_q[3] = rot_q[(face_id * 4 + 3) * n_blendshape + i_b]; // Fx4xN
                if(add_identity){ // regard q as residual, we should add quat(1,0,0,0)
                    loc_q[0] += 1;
                }
                quaternion_to_matrix(loc_mat, loc_q);
                Construct_SH_Rotation_Matrix(loc_mat, sh1_, sh2_, sh3_);
                // rot(qi,SH) - SH
                {
                    #pragma unroll
                    for(int i = 0; i < 3; i++){
                        sh1_[i][i] -= 1;
                    }
                    #pragma unroll
                    for(int i = 0; i < 5; i++){
                        sh2_[i][i] -= 1;
                    }
                    #pragma unroll
                    for(int i = 0; i < 7; i++){
                        sh3_[i][i] -= 1;
                    }
                }
                //
                MADRot3x3(sh1,sh1_,loc_grid_w[j] * k_expr);
                MADRot5x5(sh2,sh2_,loc_grid_w[j] * k_expr);
                MADRot7x7(sh3,sh3_,loc_grid_w[j] * k_expr);
            }
        }
        // finally we add mean to transform matrix
        {
            #pragma unroll
            for(int i = 0; i < 3; i++){
                sh1[i][i] += 1;
            }
            #pragma unroll
            for(int i = 0; i < 5; i++){
                sh2[i][i] += 1;
            }
            #pragma unroll
            for(int i = 0; i < 7; i++){
                sh3[i][i] += 1;
            }
        }
        // apply rot matrix to SH
        loc_sh_out[0] = loc_sh[0];

        loc_sh_out[1] = sh1[0][0] * loc_sh[1] + sh1[0][1] * loc_sh[2] + sh1[0][2] * loc_sh[3];
		loc_sh_out[2] = sh1[1][0] * loc_sh[1] + sh1[1][1] * loc_sh[2] + sh1[1][2] * loc_sh[3];
		loc_sh_out[3] = sh1[2][0] * loc_sh[1] + sh1[2][1] * loc_sh[2] + sh1[2][2] * loc_sh[3];

		#pragma unroll
		for (int i = 0; i < 5; i++) {
			loc_sh_out[4 + i] = 0;
#pragma unroll
			for (int j = 0; j < 5; j++) {
				loc_sh_out[4 + i] += sh2[i][j] * loc_sh[4 + j];
			}
		}

#pragma unroll
		for (int i = 0; i < 7; i++) {
			loc_sh_out[9 + i] = 0;
#pragma unroll
			for (int j = 0; j < 7; j++) {
				loc_sh_out[9 + i] += sh3[i][j] * loc_sh[9 + j];
			}
		}

#ifdef NEG_SH_SIGN
#pragma unroll
        for (int i = 0; i < 16; i++) {
            if (i % 2) loc_sh_out[i] = -loc_sh_out[i];
        }
#endif
	}
	//// write back, first to shared memory, then to global memory
	__syncthreads();
	if(i_elem < n_elements){
#pragma unroll
		for (int i = 0; i < 16; i++) {
			s_shs[li_elem * 48 + i * 3 + channel] = loc_sh_out[i];
		}
	}
	__syncthreads();
#pragma unroll
	for (int i = 0; i < 16; i++) {
		int loc_offset = i * 96 + threadIdx.x;
		int64_t i_elem = blockIdx.x * (96 * 16) + loc_offset;
		if (i_elem < n_elements * 48) {
			shs_out[i_elem] = s_shs[loc_offset];
		}
	}
}


template<int SN>
__global__ void __launch_bounds__(64) CompositeBlendShapeKernel(
    int64_t P, int64_t M, int64_t K,
    float * __restrict__ output, // PxM
    const float* __restrict__ mean, // PxM
    const float* __restrict__ blendshape, // PxMxK
    const float* __restrict__ coefficient, // K
    const float* __restrict__ init_blendshape, // PxMxK
    const float* __restrict__ mask // PxK
){
    __shared__ float s_buff[SN * 64];
    __shared__ float s_coeff[SN];
    if(threadIdx.x < SN){
        s_coeff[threadIdx.x] = coefficient[threadIdx.x];
    }
    const int n_load_pass = (SN * 64 + 63) / 64;
    int64_t i_elem = blockIdx.x * 64 + threadIdx.x;
    float v = 0;
    if(init_blendshape){
        for(int i = 0; i < n_load_pass ; i++){
            int loc_offset = i * 64 + threadIdx.x;
            int64_t i_elem = (int64_t)(blockIdx.x * 64) * SN + loc_offset;
            if(i_elem < SN * (P * M) && loc_offset < SN * 64){
                s_buff[loc_offset] = init_blendshape[i_elem];
            }
        }
        __syncthreads(); // make sure s_coeff & s_buffer is correctly loaded
        if(i_elem < P * M){
            for(int ii = 0; ii < SN; ii++){
                float k = s_coeff[ii];
                float bt = s_buff[threadIdx.x * SN + ii];
                v += bt * k;
            }
        }
        __syncthreads(); // make sure all done, otherwise s_buff maybe overwritten
    }
    for(int i = 0; i < n_load_pass; i++){
        int loc_offset = i * 64 + threadIdx.x;
        int64_t i_elem = (int64_t)(blockIdx.x * 64) * SN + loc_offset;
        if(i_elem < SN * (P * M) && loc_offset < SN * 64){
            s_buff[loc_offset] = blendshape[i_elem];
        }
    }
    __syncthreads();
    if(i_elem < P * M){
        v += mean[i_elem];
        for(int ii = 0; ii < SN; ii++){
            float k = s_coeff[ii];
            float bt = s_buff[threadIdx.x * SN + ii];
            if(mask){
                int64_t i_p = i_elem / M;
                bt = bt * mask[i_p * SN + ii];
            }
            v += bt * k;
        }
        output[i_elem] = v;
    }
}

template<int SN>
__global__ void __launch_bounds__(64) CompositeBlendShapeKernel_Backward(
    int64_t P, int64_t M, int64_t K,
    const float * __restrict__ grad_out, // PxM
    float * __restrict__ grad_mean, // PxM
    float * __restrict__ grad_blendshape, // PxMxK
    const float * __restrict__ coefficient, // K
    const float * __restrict__ mask // PxK
){
    __shared__ float s_buff[SN * 64];
    __shared__ float s_coeff[SN];
    if(threadIdx.x < SN){
        s_coeff[threadIdx.x] = coefficient[threadIdx.x];
    }
    __syncthreads();
    const int n_load_pass = (SN * 64 + 63) / 64;
    int64_t i_elem = blockIdx.x * 64 + threadIdx.x;
    if(i_elem < P * M){
        float grad_v = grad_out[i_elem];
        grad_mean[i_elem] = grad_v; // directly copy
        for(int ii = 0; ii < SN; ii++){
            float k = s_coeff[ii];
            if(mask){
                int64_t i_p = i_elem / M;
                k = k * mask[i_p * SN + ii];
            }
            s_buff[threadIdx.x * SN + ii] = k * grad_v;
        }
    }
    __syncthreads(); // write shared memory to global
    for(int i = 0; i < n_load_pass; i++){
        int loc_offset = i * 64 + threadIdx.x;
        int64_t i_elem = (int64_t)(blockIdx.x * 64) * SN + loc_offset;
        if(i_elem < SN * (P * M) && loc_offset < SN * 64){
            grad_blendshape[i_elem] = s_buff[loc_offset];
        }
    }
}

__global__ void __launch_bounds__(128) AdamUpdateKernel(
    int64_t N, float * __restrict__ param,  float * __restrict__ grad,
    float * __restrict__ exp_avg, float * __restrict__ exp_avg_sq,
    float beta1, float beta2, float bias_correction1, float sqrt_bias_correction2,
    float step_size, float eps, float weight_decay
){
    int64_t i_elem = (int64_t)blockIdx.x * 128 + threadIdx.x;
    if(i_elem < N){
        float grad_ = grad[i_elem];
        float param_ = param[i_elem];
        if (weight_decay != 0){
            grad_ = grad_ + param_ * weight_decay;
        }
        float exp_avg_ = exp_avg[i_elem] * beta1 + (1-beta1) * grad_;
        float exp_avg_sq_ = exp_avg_sq[i_elem] * beta2 + (1-beta2) * (grad_ * grad_);
        float denom = sqrtf(exp_avg_sq_) / sqrt_bias_correction2 + eps;
        param_ -= step_size * exp_avg_ / denom;
        // write back
        exp_avg[i_elem] = exp_avg_;
        exp_avg_sq[i_elem] = exp_avg_sq_;
        param[i_elem] = param_;
    }
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


torch::Tensor RotSH3_C3(
    const torch::Tensor & rot, // Bx3x3
    const torch::Tensor & sh // Bx16x3
){
    CHECK_CUDA(rot); CHECK_CUDA(sh);
    CHECK_CONTIGUOUS(rot); CHECK_CONTIGUOUS(sh);
    CHECK_IS_FLOATING(rot); CHECK_IS_FLOATING(sh);

    int64_t B = rot.numel() / 9;
    int64_t B2 = sh.numel() / (16 * 3);
    TORCH_CHECK(B == B2, "Batch size must match");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(sh));
    auto device = sh.device();

    at::TensorOptions opt(sh.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);

    torch::Tensor sh_out = torch::empty({B, 16, 3}, opt);

    const uint32_t THREADS = 32 * 3;
    const uint32_t BLOCKS = (uint32_t)div_round_up(B, (int64_t)32);

    if(BLOCKS){
        CompositeBasisRotSHKernel<<<BLOCKS,THREADS>>>(
            B, rot.data_ptr<float>(),
            sh.data_ptr<float>(),
            sh_out.data_ptr<float>()
        );
    }

    return sh_out;
}


torch::Tensor RotSH3_C3_simple_backward(
    const torch::Tensor & rot, // Bx3x3
    const torch::Tensor & grad // Bx16x3
){
    CHECK_CUDA(rot); CHECK_CUDA(grad);
    CHECK_CONTIGUOUS(rot); CHECK_CONTIGUOUS(grad);
    CHECK_IS_FLOATING(rot); CHECK_IS_FLOATING(grad);

    int64_t B = rot.numel() / 9;
    int64_t B2 = grad.numel() / (16 * 3);
    TORCH_CHECK(B == B2, "Batch size must match");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad));
    auto device = grad.device();

    at::TensorOptions opt(grad.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);
    torch::Tensor grad_out = torch::empty({B, 16, 3}, opt);

    const uint32_t THREADS = 32 * 3;
    const uint32_t BLOCKS = (uint32_t)div_round_up(B, (int64_t)32);

    if(BLOCKS){
        CompositeBasisRotSHKernel_BackwardSimple<<<BLOCKS,THREADS>>>(
            B, rot.data_ptr<float>(),
            grad.data_ptr<float>(),
            grad_out.data_ptr<float>()
        );
    }
    return grad_out;
}

std::tuple<torch::Tensor, torch::Tensor> RotSH3_C3_backward(
    const torch::Tensor & rot, // Bx3x3
    const torch::Tensor & sh, // Bx16x3
    const torch::Tensor & grad // Bx16x3
){
    CHECK_CUDA(rot); CHECK_CUDA(sh); CHECK_CUDA(grad);
    CHECK_CONTIGUOUS(rot); CHECK_CONTIGUOUS(sh); CHECK_CONTIGUOUS(grad);
    CHECK_IS_FLOATING(rot); CHECK_IS_FLOATING(sh); CHECK_IS_FLOATING(grad);

    int64_t B = rot.numel() / (3 * 3);
    int64_t B1 = sh.numel() / (16 * 3);
    int64_t B2 = grad.numel() / (16 * 3);
    TORCH_CHECK(B == B2 && B == B1, "Batch size must match");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad));
    auto device = grad.device();

    at::TensorOptions opt(grad.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);

    torch::Tensor sh_grad = torch::empty({B, 16, 3}, opt);
    torch::Tensor rot_grad = torch::empty({B, 3, 3}, opt);

    const uint32_t THREADS = 32 * 3;
    const uint32_t BLOCKS = (uint32_t)div_round_up(B, (int64_t)32);

    if(BLOCKS){
        CompositeBasisRotSHKernel_Backward<<<BLOCKS,THREADS>>>(
            B, rot.data_ptr<float>(),
            sh.data_ptr<float>(),
            grad.data_ptr<float>(),
            rot_grad.data_ptr<float>(),
            sh_grad.data_ptr<float>()
        );
    }

    return {rot_grad, sh_grad};
}

torch::Tensor RotSHBlendShape(
    const torch::Tensor & expr, // N
    const torch::Tensor & rot_q, // Fx4xN
    const torch::Tensor & grid_id, // Px8
    const torch::Tensor & grid_w, // Px8
    const torch::Tensor & sh // Px16x3
){
    CHECK_CUDA(expr); CHECK_CUDA(rot_q); CHECK_CUDA(grid_id); CHECK_CUDA(grid_w); CHECK_CUDA(sh);
    CHECK_CONTIGUOUS(expr); CHECK_CONTIGUOUS(rot_q); CHECK_CONTIGUOUS(grid_id); CHECK_CONTIGUOUS(grid_w);CHECK_CONTIGUOUS(sh);
    CHECK_IS_FLOATING(expr); CHECK_IS_FLOATING(rot_q); CHECK_IS_FLOATING(grid_w); CHECK_IS_FLOATING(sh);
    CHECK_IS_INT(grid_id);
    // TODO, shape check

    int64_t B = sh.numel() / (16 * 3);
    int64_t N = expr.numel();

    const at::cuda::OptionalCUDAGuard device_guard(device_of(sh));
    auto device = sh.device();

    at::TensorOptions opt(sh.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);
    torch::Tensor sh_out = torch::empty({B, 16, 3}, opt);

    const uint32_t THREADS = 32 * 3;
    const uint32_t BLOCKS = (uint32_t)div_round_up(B, (int64_t)32);

    if(BLOCKS){
        RotSHBlendShapeKernel<<<BLOCKS,THREADS>>>(
            B,N,
            expr.data_ptr<float>(),
            rot_q.data_ptr<float>(),
            grid_id.data_ptr<int>(),
            grid_w.data_ptr<float>(),
            sh.data_ptr<float>(),
            sh_out.data_ptr<float>(),
            true
        );
    }
    return sh_out;
}

torch::Tensor RotSHBlendShape_backward(
    const torch::Tensor & expr, // N
    const torch::Tensor & rot_q, // Fx4xN
    const torch::Tensor & grid_id, // Px8
    const torch::Tensor & grid_w, // Px8
    const torch::Tensor & grad // Px16x3
){
    CHECK_CUDA(expr); CHECK_CUDA(rot_q); CHECK_CUDA(grid_id); CHECK_CUDA(grid_w); CHECK_CUDA(grad);
    CHECK_CONTIGUOUS(expr); CHECK_CONTIGUOUS(rot_q); CHECK_CONTIGUOUS(grid_id); CHECK_CONTIGUOUS(grid_w);CHECK_CONTIGUOUS(grad);
    CHECK_IS_FLOATING(expr); CHECK_IS_FLOATING(rot_q); CHECK_IS_FLOATING(grid_w); CHECK_IS_FLOATING(grad);
    CHECK_IS_INT(grid_id);
    // TODO, shape check

    int64_t B = grad.numel() / (16 * 3);
    int64_t N = expr.numel();

    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad));
    auto device = grad.device();

    at::TensorOptions opt(grad.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);
    torch::Tensor grad_out = torch::empty({B, 16, 3}, opt);

    const uint32_t THREADS = 32 * 3;
    const uint32_t BLOCKS = (uint32_t)div_round_up(B, (int64_t)32);

    if(BLOCKS){
        RotSHBlendShapeKernel_Backward<<<BLOCKS,THREADS>>>(
            B,N,
            expr.data_ptr<float>(),
            rot_q.data_ptr<float>(),
            grid_id.data_ptr<int>(),
            grid_w.data_ptr<float>(),
            grad.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            true
        );
    }
    return grad_out;
}

torch::Tensor InvMat3x3(
    const torch::Tensor & mat // Bx3x3
){
    CHECK_CUDA(mat); CHECK_CONTIGUOUS(mat); CHECK_IS_FLOATING(mat);
    int64_t B = mat.numel() / 9; // TODO, check dimension

    const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
    auto device = mat.device();

    at::TensorOptions opt(mat.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);
    torch::Tensor mat_out = torch::empty({B, 3, 3}, opt);
    const uint32_t THREADS = 128;
    const uint32_t BLOCKS = (uint32_t)div_round_up(B, (int64_t)128);

    if(BLOCKS){
        InvMat3x3Kernel<<<BLOCKS,THREADS>>>(
            B,
            mat.data_ptr<float>(),
            mat_out.data_ptr<float>()
        );
    }

    return mat_out;
}

torch::Tensor InvMat3x3_backward(
    const torch::Tensor & mat, // Bx3x3
    const torch::Tensor & grad_out // Bx3x3
){
    CHECK_CUDA(mat); CHECK_CONTIGUOUS(mat); CHECK_IS_FLOATING(mat);
    CHECK_CUDA(grad_out); CHECK_CONTIGUOUS(grad_out); CHECK_IS_FLOATING(grad_out);

    int64_t B = mat.numel() / 9;
    int64_t B2 = grad_out.numel() / 9;
    TORCH_CHECK(B == B2, "Batch size must match");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
    auto device = mat.device();

    at::TensorOptions opt(mat.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);
    torch::Tensor grad_in = torch::empty({B,3,3},opt);
    const uint32_t THREADS = 128;
    const uint32_t BLOCKS = (uint32_t)div_round_up(B, (int64_t)128);

    if(BLOCKS){
        InvMat3x3Kernel_Backward<<<BLOCKS,THREADS>>>(
            B,
            mat.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            grad_in.data_ptr<float>()
        );
    }
    return grad_in;
}

torch::Tensor CompositeBlendShape(
    const torch::Tensor & mean, // Px?
    const torch::Tensor & blendshape, // Px?xK
    const torch::Tensor & coefficient, // K
    at::optional<at::Tensor> & init_blendshape, // Px?xK
    at::optional<at::Tensor> & mask // PxK
){
    CHECK_CUDA(mean); CHECK_CONTIGUOUS(mean); CHECK_IS_FLOATING(mean);
    CHECK_CUDA(blendshape); CHECK_CONTIGUOUS(blendshape); CHECK_IS_FLOATING(blendshape);
    CHECK_CUDA(coefficient); CHECK_CONTIGUOUS(coefficient); CHECK_IS_FLOATING(coefficient);

    const float * ptr_init_blendshape = nullptr;
    const float * ptr_mask = nullptr;

    bool use_init_blendshape = init_blendshape.has_value();
    bool use_mask = mask.has_value();

    if(use_init_blendshape){
        at::Tensor tmp = init_blendshape.value();
        CHECK_CUDA(tmp); CHECK_CONTIGUOUS(tmp); CHECK_IS_FLOATING(tmp);
        ptr_init_blendshape = tmp.data_ptr<float>();
    }
    if(use_mask){
        at::Tensor tmp = mask.value();
        CHECK_CUDA(tmp); CHECK_CONTIGUOUS(tmp); CHECK_IS_FLOATING(tmp);
        ptr_mask = tmp.data_ptr<float>();
    }

    // TODO, shape validity check
    int64_t K = blendshape.size(-1);
    int64_t P = mean.size(0);
    int64_t M = mean.numel() / P;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(mean));
    auto device = mean.device();
    at::TensorOptions opt(mean.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);
    torch::Tensor composite_output = torch::empty({P,M},opt); // view outside !

    const uint32_t THREADS = 64;
    const uint32_t BLOCKS = (uint32_t)div_round_up(P * M, (int64_t)64);
    if(BLOCKS){
        if(K == 50){
            CompositeBlendShapeKernel<50><<<BLOCKS,THREADS>>>(
                P,M,K,
                composite_output.data_ptr<float>(),
                mean.data_ptr<float>(),
                blendshape.data_ptr<float>(),
                coefficient.data_ptr<float>(),
                ptr_init_blendshape,
                ptr_mask
            );
        }
        else if (K == 51){ // FaceUnity blendshape
            CompositeBlendShapeKernel<51><<<BLOCKS,THREADS>>>(
                P,M,K,
                composite_output.data_ptr<float>(),
                mean.data_ptr<float>(),
                blendshape.data_ptr<float>(),
                coefficient.data_ptr<float>(),
                ptr_init_blendshape,
                ptr_mask
            );
        }
        else{
            throw std::runtime_error("Blendshape number not supported!");
        }
    }
    return composite_output;
}

std::vector<torch::Tensor> CompositeBlendShape_backward(
    const torch::Tensor & grad_out, // Px?
    const torch::Tensor & coefficient, // K
    at::optional<at::Tensor> & mask // PxK
){
    CHECK_CUDA(grad_out); CHECK_CONTIGUOUS(grad_out); CHECK_IS_FLOATING(grad_out);
    CHECK_CUDA(coefficient); CHECK_CONTIGUOUS(coefficient); CHECK_IS_FLOATING(coefficient);

    const float * ptr_mask = nullptr;
    bool use_mask = mask.has_value();

    if(use_mask){
        at::Tensor tmp = mask.value();
        CHECK_CUDA(tmp); CHECK_CONTIGUOUS(tmp); CHECK_IS_FLOATING(tmp);
        ptr_mask = tmp.data_ptr<float>();
    }

    // TODO, shape validity check
    int64_t K = coefficient.size(-1);
    int64_t P = grad_out.size(0);
    int64_t M = grad_out.numel() / P;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_out));
    auto device = grad_out.device();
    at::TensorOptions opt(grad_out.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);
    torch::Tensor grad_mean = torch::empty({P, M}, opt);
    torch::Tensor grad_blendshape = torch::empty({P, M, K}, opt);

    const uint32_t THREADS = 64;
    const uint32_t BLOCKS = (uint32_t)div_round_up(P * M, (int64_t)64);
    if(BLOCKS){
        if (K == 50){
            CompositeBlendShapeKernel_Backward<50><<<BLOCKS,THREADS>>>(
                P, M, K,
                grad_out.data_ptr<float>(),
                grad_mean.data_ptr<float>(),
                grad_blendshape.data_ptr<float>(),
                coefficient.data_ptr<float>(),
                ptr_mask
            );
        }
        else if (K == 51){
            CompositeBlendShapeKernel_Backward<51><<<BLOCKS,THREADS>>>(
                P, M, K,
                grad_out.data_ptr<float>(),
                grad_mean.data_ptr<float>(),
                grad_blendshape.data_ptr<float>(),
                coefficient.data_ptr<float>(),
                ptr_mask
            );
        }
        else{
            throw std::runtime_error("Blendshape number not supported!");
        }
    }
    return {grad_mean, grad_blendshape};
}

void AdamUpdate(
    float beta1, float beta2, float bias_correction1, float sqrt_bias_correction2,
    float step_size,  float eps, float weight_decay,
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq
){
    CHECK_CUDA(param); CHECK_CONTIGUOUS(param); CHECK_IS_FLOATING(param);
    CHECK_CUDA(grad); CHECK_CONTIGUOUS(grad); CHECK_IS_FLOATING(grad);
    CHECK_CUDA(exp_avg); CHECK_CONTIGUOUS(exp_avg); CHECK_IS_FLOATING(exp_avg);
    CHECK_CUDA(exp_avg_sq); CHECK_CONTIGUOUS(exp_avg_sq); CHECK_IS_FLOATING(exp_avg_sq);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(param));

    const int64_t N = param.numel();
    const uint32_t THREADS = 128;
    const uint32_t BLOCKS = (uint32_t)div_round_up(N, (int64_t)128);
    if(BLOCKS){
        AdamUpdateKernel<<<BLOCKS,THREADS>>>(
            N, param.data_ptr<float>(), grad.data_ptr<float>(),
            exp_avg.data_ptr<float>(), exp_avg_sq.data_ptr<float>(),
            beta1, beta2, bias_correction1, sqrt_bias_correction2,
            step_size, eps, weight_decay
        );
    }
}