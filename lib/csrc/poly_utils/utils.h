#pragma once
#include <torch/extension.h>


std::tuple<at::Tensor, at::Tensor> calculate_corners(
    const at::Tensor& s_ids,
    const at::Tensor& e_ids,
    const at::Tensor& s_e_num,
    const at::Tensor& start_idx,
    const at::Tensor& poly_edge_len,
    const at::Tensor& p_edge_num,
    const at::Tensor& poly_num,
    const int max_num
);

std::tuple<at::Tensor, at::Tensor> calculate_wnp_iter(
    const at::Tensor& edge_num,
    const at::Tensor& edge_start_idx,
    const at::Tensor& poly_num,
    const int p_num
);