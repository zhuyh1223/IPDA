#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
//#include <THC/THC.h>
//#include <THC/THCAtomics.cuh>
//#include <THC/THCDeviceUtils.cuh>
#include "cuda_common.h"

__global__ void _calculate_corners(
    const int64_t* s_ids,
    const int64_t* e_ids,
    const int64_t* s_e_num,
    const int64_t* start_idx,
    const float* poly_edge_len,
    const int64_t* p_edge_num,
    const int64_t* poly_num,
    int64_t* ind,
    int64_t* edge_num,
    const int b,
    const int n,
    const int p_num,
    const int max_num
) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= b * n * p_num)
        return;

    const int c_b = index / (n * p_num);
    const int c_n = (index - c_b * n * p_num) / p_num;
    const int c_p_idx = index % p_num;

    const int c_s_id = s_ids[index];
    const int c_e_id = e_ids[index];
    const int c_s_e_num = s_e_num[index];
    const int c_start_idx = start_idx[index];
    if (c_s_id == c_e_id == c_start_idx == -1)
        return;

    const int c_p_edge_num = p_edge_num[index];
    const int c_poly_num = poly_num[c_b * n + c_n];
    int64_t* c_ind = &ind[c_b * n * max_num + c_n * max_num + c_start_idx];
    int64_t* c_edge_num = &edge_num[c_b * n * max_num + c_n * max_num + c_start_idx];

    float edge_len_sum = 0;
    for (long i = 0; i < c_s_e_num; i++) {
        c_ind[i] = long((c_s_id + i) % c_poly_num);
        if (i == c_s_e_num - 1){
            if (c_ind[i] != (c_e_id - 1 + c_poly_num) % c_poly_num)
                c_ind[i] = (c_e_id - 1 + c_poly_num) % c_poly_num;
        }
        edge_len_sum = edge_len_sum + poly_edge_len[c_b * n * p_num + c_n * p_num + c_ind[i]];
    }

    int edge_num_sum = 0;
    float edge_len_i = 0;
    for (long i = 0; i < c_s_e_num; i++) {
        edge_len_i = poly_edge_len[c_b * n * p_num + c_n * p_num + c_ind[i]];
        c_edge_num[i] = int(round(float(edge_len_i) * float(c_p_edge_num) / float(edge_len_sum)));
        if (c_edge_num[i] == 0)
            c_edge_num[i] = 1;
        edge_num_sum = edge_num_sum + c_edge_num[i];
    }

    if (edge_num_sum == c_p_edge_num)
        return;

    if (edge_num_sum < c_p_edge_num)
        c_edge_num[0] += c_p_edge_num - edge_num_sum;
    else {
        int id = 0;
        long pass_num = edge_num_sum - c_p_edge_num;
        while (pass_num > 0) {
            if (c_edge_num[id] > pass_num) {
                c_edge_num[id] -= pass_num;
                pass_num = 0;
            } else if (c_edge_num[id] == 1) {
                id += 1;
            } else {
                pass_num -= c_edge_num[id] - 1;
                c_edge_num[id] = 1;
                id += 1;
            }
        }
    }
}

std::tuple<at::Tensor, at::Tensor> calculate_corners(
    const at::Tensor& s_ids,
    const at::Tensor& e_ids,
    const at::Tensor& s_e_num,
    const at::Tensor& start_idx,
    const at::Tensor& poly_edge_len,
    const at::Tensor& p_edge_num,
    const at::Tensor& poly_num,
    const int max_num
) {
    AT_ASSERTM(s_ids.type().is_cuda(), "s_ids must be a CUDA tensor");
    AT_ASSERTM(e_ids.type().is_cuda(), "e_ids must be a CUDA tensor");
    AT_ASSERTM(s_e_num.type().is_cuda(), "s_e_num must be a CUDA tensor");
    AT_ASSERTM(start_idx.type().is_cuda(), "start_idx must be a CUDA tensor");
    AT_ASSERTM(poly_edge_len.type().is_cuda(), "poly_edge_len must be a CUDA tensor");
    AT_ASSERTM(p_edge_num.type().is_cuda(), "p_edge_num must be a CUDA tensor");
    AT_ASSERTM(poly_num.type().is_cuda(), "poly_num must be a CUDA tensor");

    // b, n, p_num
    auto b = s_ids.size(0);
    auto n = s_ids.size(1);
    auto p_num = s_ids.size(2);

    // e_ids b, n, p_num
    AT_ASSERTM(b == e_ids.size(0), "e_ids must have the same batch size with s_ids");
    AT_ASSERTM(n == e_ids.size(1), "e_ids must have the same instances size with s_ids");
    AT_ASSERTM(p_num == e_ids.size(2), "e_ids must have the same points size with s_ids");

    // s_e_num b, n, p_num
    AT_ASSERTM(b == s_e_num.size(0), "s_e_num must have the same batch size with s_ids");
    AT_ASSERTM(n == s_e_num.size(1), "s_e_num must have the same instances size with s_ids");
    AT_ASSERTM(p_num == s_e_num.size(2), "s_e_num must have the same points size with s_ids");

    // start_idx b, n, p_num
    AT_ASSERTM(b == start_idx.size(0), "start_idx must have the same batch size with s_ids");
    AT_ASSERTM(n == start_idx.size(1), "start_idx must have the same instances size with s_ids");
    AT_ASSERTM(p_num == start_idx.size(2), "start_idx must have the same points size with s_ids");

    // poly_edge_len b, n, p_num
    AT_ASSERTM(b == poly_edge_len.size(0), "poly_edge_len must have the same batch size with s_ids");
    AT_ASSERTM(n == poly_edge_len.size(1), "poly_edge_len must have the same instances size with s_ids");
    AT_ASSERTM(p_num == poly_edge_len.size(2), "poly_edge_len must have the same points size with s_ids");

    // p_edge_num b, n, p_num
    AT_ASSERTM(b == p_edge_num.size(0), "p_edge_num must have the same batch size with s_ids");
    AT_ASSERTM(n == p_edge_num.size(1), "p_edge_num must have the same instances size with s_ids");
    AT_ASSERTM(p_num == p_edge_num.size(2), "p_edge_num must have the same points size with s_ids");

    // poly_num b, n
    AT_ASSERTM(b == poly_num.size(0), "poly_num must have the same batch size with s_ids");
    AT_ASSERTM(n == poly_num.size(1), "poly_num must have the same instances size with s_ids");

    auto ind = at::full({b, n, max_num}, -1, p_edge_num.options());
    auto edge_num = at::zeros({b, n, max_num}, p_edge_num.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int bdim0, bdim1, bdim2;
    int tdim0, tdim1, tdim2;
    getGPULayout(b * n * p_num, 1, 1, &bdim0, &bdim1, &bdim2, &tdim0, &tdim1, &tdim2);
    dim3 bdim(bdim0, bdim1, bdim2);
    dim3 tdim(tdim0, tdim1, tdim2);

    _calculate_corners<<<bdim, tdim, 0, stream>>>(
        s_ids.contiguous().data<int64_t>(),
        e_ids.contiguous().data<int64_t>(),
        s_e_num.contiguous().data<int64_t>(),
        start_idx.contiguous().data<int64_t>(),
        poly_edge_len.contiguous().data<float>(),
        p_edge_num.contiguous().data<int64_t>(),
        poly_num.contiguous().data<int64_t>(),
        ind.data<int64_t>(),
        edge_num.data<int64_t>(),
        b,
        n,
        p_num,
        max_num
    );
    C10_CUDA_CHECK(cudaGetLastError());

    return std::make_tuple(ind, edge_num);
}


__global__ void _calculate_wnp_iter(
    const int64_t* edge_num,
    const int64_t* edge_start_idx,
    const int64_t* poly_num,
    float* weight,
    int64_t* ind,
    const int b,
    const int n,
    const int orig_p_num,
    const int p_num
) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= b * n * orig_p_num)
        return;

    const int c_b = index / (n * orig_p_num);
    const int c_n = (index - c_b * n * orig_p_num) / orig_p_num;
    const int c_edge_idx = index % orig_p_num;

    const long c_edge_num = edge_num[index];
    const int c_start_idx = int(edge_start_idx[index]);
    const int c_poly_num = poly_num[c_b * n + c_n];
    if (c_start_idx >= p_num)
        return;
    float* c_weight = &weight[c_b * n * p_num + c_n * p_num + c_start_idx];
    int64_t* c_ind = &ind[c_b * n * p_num * 2 + c_n * p_num * 2 + c_start_idx * 2];

    for (long i = 0; i < c_edge_num; i++) {
        c_weight[i] = float(i) / float(c_edge_num);
        c_ind[i * 2] = long(c_edge_idx);
        c_ind[i * 2 + 1] = long((c_edge_idx + 1) % c_poly_num);
    }
}

std::tuple<at::Tensor, at::Tensor> calculate_wnp_iter(
    const at::Tensor& edge_num,
    const at::Tensor& edge_start_idx,
    const at::Tensor& poly_num,
    const int p_num
) {
    AT_ASSERTM(edge_num.type().is_cuda(), "edge_num must be a CUDA tensor");
    AT_ASSERTM(edge_start_idx.type().is_cuda(), "edge_start_idx must be a CUDA tensor");
    AT_ASSERTM(poly_num.type().is_cuda(), "poly_num must be a CUDA tensor");

    // b, n, orig_p_num
    auto b = edge_num.size(0);
    auto n = edge_num.size(1);
    auto orig_p_num = edge_num.size(2);

    // edge_start_idx b, n, orig_p_num
    AT_ASSERTM(b == edge_start_idx.size(0), "edge_start_idx must have the same batch size with edge_num");
    AT_ASSERTM(n == edge_start_idx.size(1), "edge_start_idx must have the same instances size with edge_num");
    AT_ASSERTM(orig_p_num == edge_start_idx.size(2), "edge_start_idx must have the same points size with edge_num");

    // poly_num b, n
    AT_ASSERTM(b == poly_num.size(0), "poly_num must have the same batch size with edge_num");
    AT_ASSERTM(n == poly_num.size(1), "poly_num must have the same instances size with edge_num");

    auto weight = at::zeros({b, n, p_num, 1}, edge_num.options().dtype(at::kFloat));
    auto ind = at::zeros({b, n, p_num, 2}, edge_num.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int bdim0, bdim1, bdim2;
    int tdim0, tdim1, tdim2;
    getGPULayout(b * n * orig_p_num, 1, 1, &bdim0, &bdim1, &bdim2, &tdim0, &tdim1, &tdim2);
    dim3 bdim(bdim0, bdim1, bdim2);
    dim3 tdim(tdim0, tdim1, tdim2);

    _calculate_wnp_iter<<<bdim, tdim, 0, stream>>>(
        edge_num.contiguous().data<int64_t>(),
        edge_start_idx.contiguous().data<int64_t>(),
        poly_num.contiguous().data<int64_t>(),
        weight.data<float>(),
        ind.data<int64_t>(),
        b,
        n,
        orig_p_num,
        p_num
    );
    C10_CUDA_CHECK(cudaGetLastError());

    return std::make_tuple(weight, ind);
}
