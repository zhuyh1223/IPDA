import torch.nn as nn
import torch
import numpy as np
from lib.utils.snake import snake_gcn_utils
from lib.csrc.extreme_utils import _ext as extreme_utils
from lib.csrc.poly_utils import _poly as _poly_utils
from lib.config import cfg

def gaussian_scores(v, sigma=(1, 1), rho=0):
    if not isinstance(sigma, tuple):
        sigma = (sigma, sigma)
    min_v = v[int((len(v) - 1) / 2)]
    x = v[:, 0] - min_v[0]
    y = v[:, 1] - min_v[1]
    sigma_x, sigma_y = sigma
    energy = (x * x) / (sigma_x * sigma_x) - 2 * rho * x * y / (sigma_x * sigma_y) + (y * y) / (sigma_y * sigma_y)
    h = np.exp(-energy / (2 * (1 - rho * rho)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def get_corner_cls2(poly, points):
    """
    :param poly: (m, 2)
    :param points: (n, 2)
    :return: cls: (m, 1)
    """
    radius = cfg.radius
    n = points.shape[0]
    m = poly.shape[0]
    m_points = np.tile(np.reshape(points, (n, 1, 2)), (1, m, 1))
    m_poly = np.tile(np.reshape(poly, (1, m, 2)), (n, 1, 1))
    m_v = m_points - m_poly
    m_d = np.linalg.norm(m_points - m_poly, axis=2)
    min_idx = np.argmin(m_d, axis=1)
    m_cls = np.zeros((n, m))
    r_ids = np.stack([(min_idx + i) % m for i in range(-radius, radius + 1)], axis=1)
    for i, m_idx in enumerate(min_idx):
        idx_i = r_ids[i]
        v_i = np.tile((idx_i - m_idx)[:, None], (1, 2))
        m_cls[i, idx_i] = gaussian_scores(v_i, sigma=(2 * radius + 1) / 1.5)
    cls = np.max(m_cls, axis=0)
    return cls[:, None]


def cls_nms(cls, radius=cfg.radius, p_01=None, py_pred=None):
    """
    :param cls: [b, n, 1]
    :param p_01: [b, n]
    :param py_pred: [b, n, 2]
    :return: nms_cls: [b, n, 1]
    """
    if len(cls) == 0:
        return cls
    if p_01 is not None:
        o_cls = (p_01[..., None] == 2).float()
        cls = torch.max(cls, o_cls)
    b, p_num = cls.size(0), cls.size(1)
    kernel = 2 * radius + 1
    pad_cls = torch.cat([cls[:, -radius:], cls, cls[:, :radius]], dim=1)
    pad_cls = pad_cls.permute(0, 2, 1)
    hmax = nn.functional.max_pool1d(pad_cls, kernel, stride=1).permute(0, 2, 1)
    keep = (hmax == cls).float()
    nms_cls = cls * keep
    nms_cls = nms_cls[:, :, 0]
    if py_pred is not None and p_01 is not None:
        thr = radius / 2
        sorted, indices = torch.sort(nms_cls, 1, descending=True)
        indices = torch.where(sorted > cfg.s, indices, torch.full_like(indices, -1))
        k_idx = (indices >= 0).nonzero()
        k_num = (indices >= 0).sum(1)
        if k_idx.size(0) > 0:
            pts = torch.zeros_like(py_pred).view(-1, 2)
            arrang_ids = torch.arange(b * p_num).to(pts.device) % p_num
            idx = (arrang_ids < k_num[:, None].repeat(1, p_num).view(-1,)).nonzero()[:, 0]
            pts[idx] = py_pred.view(-1, 2)[k_idx[:, 0] * p_num + indices.view(-1,)[k_idx[:, 0] * p_num + k_idx[:, 1]]]
            pts = pts.view(b, p_num, 2)
            m_pts1 = pts[:, :, None].repeat(1, 1, p_num, 1)
            m_pts2 = pts[:, None].repeat(1, p_num, 1, 1)
            m_v = m_pts1 - m_pts2
            m_d = m_v.pow(2).sum(3).sqrt()
            m_d = torch.where(m_pts1.abs().sum(3) * m_pts2.abs().sum(3) == 0, torch.full_like(m_d, -1), m_d)
            idx = ((m_d < thr).int() * (m_d != -1).int()).nonzero()
            idx_i = (idx[:, 1] != idx[:, 2]).nonzero()[:, 0]
            idx = idx[idx_i]
            idx[:, 1:] = idx[:, 0:1].repeat(1, 2) * p_num + idx[:, 1:]
            del_idx = [-1]
            idx_np = idx.detach().cpu().numpy()
            for i in range(len(idx_np)):
                if idx_np[i, 1] in del_idx:
                    continue
                del_idx.append(idx_np[i, 2])
            if len(del_idx) > 1:
                del_idx = np.stack(del_idx[1:], 0)
                del_idx = torch.tensor(del_idx).to(nms_cls.device)
                nms_cls = nms_cls.view(-1, )
                nms_cls[indices.view(-1, )[del_idx]] = 0
    return nms_cls.view(b, p_num, 1)


def add_pred_cls(pred_idx):
    """
    :param pred_idx: (b, n), -1 pad
    :return: n_idx: (b, n)
    """
    p_num = pred_idx.size(1)
    n_idx = pred_idx.clone()

    pred_num = (pred_idx >= 0).float().sum(1)
    # pred_num == 0
    idx_0 = (pred_num == 0).nonzero()[:, 0]

    if idx_0.size(0) > 0:
        n_idx[idx_0, 0] = torch.full_like(idx_0, 0)
        n_idx[idx_0, 1] = torch.full_like(idx_0, int(p_num / 3))
        n_idx[idx_0, 2] = torch.full_like(idx_0, int(2 * p_num / 3))
    # pred_num == 1
    idx_1 = (pred_num == 1).nonzero()[:, 0]
    if idx_1.size(0) > 0:
        n_idx[idx_1, 0] = pred_idx[idx_1, 0]
        n_idx[idx_1, 1] = (n_idx[idx_1, 0] + int(p_num / 3)) % p_num
        n_idx[idx_1, 2] = (n_idx[idx_1, 0] + int(2 * p_num / 3)) % p_num
        n_idx[idx_1, :3] = n_idx[idx_1, :3].sort(1)[0]
    # pred_num == 2
    idx_2 = (pred_num == 2).nonzero()[:, 0]
    if idx_2.size(0) > 0:
        n_idx[idx_2, :2] = pred_idx[idx_2, :2]
        t_idx = n_idx[idx_2, 1] - n_idx[idx_2, 0]
        t_idx = torch.where(t_idx < p_num / 2,
                            torch.round((n_idx[idx_2, 1] + n_idx[idx_2, 0] + p_num).float() / 2).long() % p_num,
                            torch.round((n_idx[idx_2, 1] + n_idx[idx_2, 0]).float() / 2).long())
        n_idx[idx_2, 2] = t_idx
        n_idx[idx_2, :3] = n_idx[idx_2, :3].sort(1)[0]

    return n_idx.type_as(pred_idx)


def random_noise(p_ids, poly, pred_match, i):
    """
    :param p_ids: (b, n) -1 pad.
    :param pred_poly: (b, n, 2)
    :param i: iter no.
    :return: n_ids: (b, n) -1 pad.
            keep_points: (b, n, 2) 0 pad.
    """
    b = p_ids.size(0)
    p_num = p_ids.size(1)
    pred_num = (p_ids >= 0).sum(1)

    pre_poly = torch.roll(poly, 1, 1)
    next_poly = torch.roll(poly, -1, 1)
    pre_poly = pre_poly.view(-1, 2)
    b_ids = torch.arange(b).to(p_ids.device)
    idx_ft1 = b_ids * p_num + pred_num % p_num
    idx_ft2 = b_ids * p_num + torch.max(pred_num - 1, torch.zeros_like(pred_num))
    pre_poly[idx_ft1] = poly.view(-1, 2)[idx_ft1]
    pre_poly[b_ids * p_num] = poly.view(-1, 2)[idx_ft2]
    pre_poly = pre_poly.view(b, p_num, 2)
    next_poly[:, -1] = poly[:, -1]
    next_poly = next_poly.view(-1, 2)
    next_poly[idx_ft2] = poly[:, 0].view(-1, 2)
    next_poly = next_poly.view(b, p_num, 2)

    pre_v = poly - pre_poly
    post_v = poly - next_poly
    max_d = torch.max(pre_v.abs(), post_v.abs())
    poly_d = torch.min(0.2 * (0.7**i) * max_d, torch.full_like(max_d, 10))
    p_shift = poly_d * torch.frac(torch.randn_like(poly_d))
    n_poly = poly + p_shift

    min_keep_num = torch.max((3 * pred_num / 4).int(), torch.full_like(pred_num.int(), 3))
    keep_num = min_keep_num + (torch.rand_like(pred_num.float()) * (pred_num.int() - min_keep_num + 1).float() - 1e-6).int()
    max_num = torch.max(pred_num)
    perm = torch.randperm(max_num * b).to(pred_num.device)
    v = (perm / max_num).int()
    m = (perm % max_num).int()
    v, indices = v.sort()
    m = m[indices]
    keep_idx = (m.long() < pred_num[:, None].repeat(1, max_num).reshape(-1,)).nonzero()[:, 0]

    arange_ids = torch.arange(b * p_num).to(p_ids.device) % p_num
    idx = (arange_ids < pred_num[:, None].repeat(1, p_num).view(-1, )).nonzero()[:, 0]
    n_ids = torch.full_like(arange_ids, 1e4)
    n_ids[idx] = m[keep_idx].long()
    n_ids = torch.where(arange_ids < keep_num[:, None].repeat(1, p_num).view(-1,).long(),
                        n_ids, torch.full_like(n_ids, 1e4))
    n_ids = n_ids.view(b, p_num)
    n_ids, _ = n_ids.sort(1)
    n_ids = torch.where(n_ids == 1e4, torch.full_like(n_ids, -1), n_ids)
    keep_points = torch.full_like(n_poly, 0).view(-1, 2)
    idx = (arange_ids < keep_num[:, None].repeat(1, p_num).view(-1,).long()).nonzero()[:, 0]
    keep_points[idx] = n_poly.view(-1, 2)[((idx / p_num).int() * p_num).long() + n_ids.view(-1,)[idx]]
    keep_points = keep_points.view(b, p_num, 2)
    n_pred_match = torch.full_like(pred_match, -1).view(-1,)
    n_pred_match[idx] = pred_match.view(-1,)[((idx / p_num).int() * p_num).long() + n_ids.view(-1,)[idx]]
    n_pred_match = n_pred_match.view(b, p_num)

    idx = (n_ids >= 0).nonzero()
    ids_ft = idx[:, 0] * p_num + idx[:, 1]
    n_ids = n_ids.view(-1,)
    n_ids[ids_ft] = p_ids.view(-1,)[idx[:, 0] * p_num + n_ids[ids_ft]]
    n_ids = n_ids.view(b, p_num)

    return n_ids, keep_points, n_pred_match


def uniform_upsample(poly, p_idx, min_v=4):
    """
    :param poly: (b, n, 2) 0 pad.
    :param p_idx: (b, n) -1 pad.
    :return:
    """
    # 1. assign point number for each edge
    # 2. calculate the coefficient for linear interpolation
    b = poly.size(0)
    p_num = poly.size(1)
    b_ids = torch.arange(b).to(p_idx.device)
    pred_num = (p_idx >= 0).sum(1)
    next_poly = torch.roll(poly, -1, 1)
    next_poly[:, -1] = poly[:, -1]
    next_poly = next_poly.view(-1, 2)
    idx_ft = b_ids * p_num + torch.max(pred_num - 1, torch.zeros_like(pred_num))
    next_poly[idx_ft] = poly[:, 0].view(-1, 2)
    next_poly = next_poly.view(b, p_num, 2)
    edge_len = (next_poly - poly).pow(2).sum(2).sqrt()
    edge_len = torch.where(p_idx == -1, torch.zeros_like(edge_len), edge_len)
    edge_num = torch.round(edge_len * p_num / torch.sum(edge_len, dim=1)[:, None]).long()
    edge_num = torch.clamp(edge_num, min=min_v)
    edge_num = torch.where(p_idx == -1, torch.zeros_like(edge_num), edge_num)
    edge_num_sum = torch.sum(edge_num, dim=1)
    edge_num = edge_num - torch.round(
        edge_num.float() * (torch.max(p_num - edge_num_sum, torch.zeros_like(edge_num_sum)) / edge_num_sum).float()[:, None]).long()

    edge_num = torch.where(p_idx == -1, torch.zeros_like(edge_num), edge_num)[None]
    edge_num_sum = torch.sum(edge_num, dim=2)
    edge_idx_sort = torch.argsort(edge_num, dim=2, descending=True)
    extreme_utils.calculate_edge_num(edge_num, edge_num_sum, edge_idx_sort, p_num)
    edge_num = torch.where(p_idx[None] == -1, torch.zeros_like(edge_num), edge_num)
    edge_num_sum = torch.sum(edge_num, dim=2)
    assert torch.all(edge_num_sum == p_num)

    edge_start_idx = torch.cumsum(edge_num, dim=2) - edge_num
    weight, ind = _poly_utils.calculate_wnp_iter(edge_num, edge_start_idx, pred_num[None], p_num)
    poly1 = poly[None].gather(2, ind[..., 0:1].expand(ind.size(0), ind.size(1), ind.size(2), 2))
    poly2 = poly[None].gather(2, ind[..., 1:2].expand(ind.size(0), ind.size(1), ind.size(2), 2))
    poly = poly1 * (1 - weight) + poly2 * weight

    return poly, edge_start_idx, edge_num


def upsample_targets(poly, s_ids, p_edge_num, gt_idx):
    """
    :param poly: (b, n, 2) 0 pad.
    :param s_ids: (b, n) -1 pad.
    :param p_edge_num: (b, n) 0 pad.
    :param gt_idx: (b, n) -1 pad.
    :return:
    """
    assert s_ids.size(0) == p_edge_num.size(0)

    b = poly.size(0)
    p_num = poly.size(1)
    b_ids = torch.arange(b).to(gt_idx.device)
    gt_num = (gt_idx >= 0).sum(1)
    # roll poly
    next_poly = torch.roll(poly, -1, 1)
    next_poly[:, -1] = poly[:, -1]
    next_poly = next_poly.view(-1, 2)
    idx_ft = b_ids * p_num + torch.max(gt_num - 1, torch.zeros_like(gt_num))
    next_poly[idx_ft] = poly[:, 0].view(-1, 2)
    next_poly = next_poly.view(b, p_num, 2)
    edge_len = (next_poly - poly).pow(2).sum(2).sqrt()
    edge_len = torch.where(gt_idx == -1, torch.zeros_like(edge_len), edge_len)
    # roll s_ids
    pred_num = (s_ids >= 0).sum(1)
    e_ids = torch.roll(s_ids, -1, 1)
    e_ids[:, -1] = s_ids[:, -1]
    e_ids = e_ids.view(-1,)
    idx_ft = b_ids * p_num + torch.max(pred_num - 1, torch.zeros_like(pred_num))
    e_ids[idx_ft] = s_ids[:, 0].view(-1, )
    e_ids = e_ids.view(b, p_num)

    s_e_num = torch.where(e_ids < s_ids, gt_num[:, None].repeat(1, p_num) + e_ids - s_ids, e_ids - s_ids)
    s_e_num = torch.clamp(s_e_num, min=1)
    s_e_num = torch.min(p_edge_num, s_e_num)
    s_e_num = torch.where(s_ids == -1, torch.full_like(s_e_num, 0), s_e_num)
    start_idx = torch.cumsum(s_e_num, dim=1) - s_e_num
    max_num = torch.max(s_e_num.sum(1))

    ind, edge_num = \
        _poly_utils.calculate_corners(s_ids[None], e_ids[None], s_e_num[None], start_idx[None],
                                      edge_len[None], p_edge_num[None], gt_num[None], max_num)

    edge_num_sum = torch.sum(edge_num, dim=2)
    assert torch.all(edge_num_sum == p_num)

    ind = ind[0]
    k_num = (ind >= 0).sum(1)
    k_idx = (ind >= 0).nonzero()
    arange_ids = torch.arange(b * p_num).to(poly.device) % p_num
    idx = (arange_ids < k_num[:, None].repeat(1, p_num).view(-1,)).nonzero()[:, 0]
    n_poly = torch.zeros_like(poly).view(-1, 2)
    n_poly[idx] = poly.view(-1, 2)[k_idx[:, 0] * p_num + ind.view(-1, )[k_idx[:, 0] * max_num + k_idx[:, 1]]]
    n_poly = n_poly.view(b, p_num, 2)

    edge_start_idx = torch.cumsum(edge_num, dim=2) - edge_num
    weight, ind = _poly_utils.calculate_wnp_iter(edge_num, edge_start_idx, k_num[None], p_num)
    poly1 = n_poly[None].gather(2, ind[..., 0:1].expand(ind.size(0), ind.size(1), ind.size(2), 2))
    poly2 = n_poly[None].gather(2, ind[..., 1:2].expand(ind.size(0), ind.size(1), ind.size(2), 2))
    up_poly = poly1 * (1 - weight) + poly2 * weight
    return up_poly, edge_start_idx


def get_gaussian(v, sigma=(1, 1), rho=0):
    """
    :param v: (n, k, 2)
    :return: h: (n, k)
    """
    if not isinstance(sigma, tuple):
        sigma = (sigma, sigma)
    min_v = v[:, int((v.size(1) - 1) / 2)]
    min_v = min_v[:, None].expand_as(v)
    x = v[..., 0] - min_v[..., 0]
    y = v[..., 1] - min_v[..., 1]
    sigma_x, sigma_y = sigma
    energy = (x * x) / (sigma_x * sigma_x) - 2 * rho * x * y / (sigma_x * sigma_y) + (y * y) / (sigma_y * sigma_y)
    h = torch.exp(-energy / (2 * (1 - rho * rho)))
    h[h < np.finfo(np.float32).eps * h.max()] = 0
    return h


def get_iter_cls(poly, iter_01):
    """
        :param poly: (b, n, 2)
        :param iter_01: (b, n)
        :return: cls: (b, n, 1)
    """
    radius = cfg.radius
    b, p_num = poly.size(0), poly.size(1)
    p_ids = (iter_01 == 1).nonzero()

    m_cls = torch.zeros([b, p_num, 1]).to(poly.device)
    if p_ids.size(0) == 0:
        return m_cls

    r_ids = torch.cat([(p_ids[:, 1:] + i) % p_num for i in range(-radius, radius + 1)], 1)
    s0, s1 = r_ids.size(0), r_ids.size(1)
    m_ids = p_ids[:, 1:].repeat(1, s1)
    r_v = ((r_ids - m_ids)[..., None]).repeat(1, 1, 2).float()
    h = get_gaussian(r_v, sigma=(2 * radius + 1) / 1.5)

    cls = torch.zeros(b * s0 * p_num).to(poly.device)
    idx = p_ids[:, :1].repeat(1, s1).view(-1,) * s0 * p_num + (h >= 0).nonzero()[:, 0] * p_num + r_ids.view(-1,)
    cls[idx] = h.view(-1,)
    cls = cls.view(b, s0, p_num, 1)
    m_cls, _ = torch.max(cls, 1)
    return m_cls


def prepare_training_iter(py_pred, cls, gt_cls, i_gt_py, iter_i):
    """
    :param py_pred: (b, n, 2)
    :param cls: (b, n, 1)
    :param gt_cls: (b, n, 1)
    :param i_gt_py: (b, n, 2)
    :return:
    """
    assert py_pred.size(0) == cls.size(0) == gt_cls.size(0) == i_gt_py.size(0)

    bt_size = i_gt_py.size(0)
    pt_num = i_gt_py.size(1)

    cls = cls[..., 0]
    gt_cls = gt_cls[..., 0]

    # prepare gt_idx_pad, gt_poly_pad
    gt_idx = (gt_cls >= 1).nonzero()
    gt_num = torch.sum((gt_cls >= 1).int(), 1)[:, None].repeat(1, pt_num)
    arange_ids = torch.arange(bt_size * pt_num).to(py_pred.device) % pt_num
    # gt_idx_pad (b * n) pad -1
    gt_idx_pad = torch.full_like(arange_ids, -1)
    idx = (arange_ids < gt_num.view(-1,)).nonzero()[:, 0]
    gt_idx_pad[idx] = gt_idx[:, 1]
    # gt_poly_pad (b * n, 2) pad 0
    gt_poly_pad = torch.zeros_like(i_gt_py).view(-1, 2)
    gt_idx_ft = gt_idx[:, 0] * pt_num + gt_idx[:, 1]
    gt_poly_pad[idx] = i_gt_py.view(-1, 2)[gt_idx_ft]

    ## remove duplicate gt vertices
    gt_poly_pad = gt_poly_pad.reshape(bt_size, pt_num, 2)
    next_poly = torch.roll(gt_poly_pad, -1, 1)
    next_poly[:, -1] = gt_poly_pad[:, -1]
    next_poly = next_poly.view(-1, 2)
    idx_ft = torch.arange(bt_size).to(gt_idx.device) * pt_num + torch.max(gt_num[:, 0] - 1, torch.zeros_like(gt_num[:, 0]))
    next_poly[idx_ft] = gt_poly_pad[:, 0].view(-1, 2)
    next_poly = next_poly.view(bt_size, pt_num, 2)
    edge_len = (next_poly - gt_poly_pad).pow(2).sum(2).sqrt()
    keep_idx = edge_len.nonzero()
    n_gt_idx = gt_idx_pad[keep_idx[:, 0] * pt_num + keep_idx[:, 1]]
    keep_num = torch.sum((edge_len > 0).int(), 1)[:, None].repeat(1, pt_num)
    gt_idx_pad = torch.full_like(arange_ids, -1)
    idx = (arange_ids < keep_num.view(-1,)).nonzero()[:, 0]
    gt_idx_pad[idx] = n_gt_idx
    gt_idx_pad = gt_idx_pad.reshape(bt_size, pt_num)
    n_gt_poly = gt_poly_pad.view(-1, 2)[keep_idx[:, 0] * pt_num + keep_idx[:, 1]]
    gt_poly_pad = torch.full_like(gt_poly_pad, 0).view(-1, 2)
    gt_poly_pad[idx] = n_gt_poly
    gt_poly_pad = gt_poly_pad.reshape(bt_size, pt_num, 2)
    b_ids = (keep_num[:, 0] == 0).nonzero()[:, 0]

    # prepare pred_idx_pad
    pred_idx = (cls > 0.3).nonzero()
    pred_num = torch.sum((cls > 0.3).int(), 1)[:, None].repeat(1, pt_num).view(-1, )
    # pred_idx_pad (b, n) pad -1
    pred_idx_pad = torch.full_like(arange_ids, -1)
    idx = (arange_ids < pred_num).nonzero()[:, 0]
    pred_idx_pad[idx] = pred_idx[:, 1]
    pred_idx_pad = pred_idx_pad.reshape(bt_size, pt_num)
    pred_idx_pad = add_pred_cls(pred_idx_pad)
    idx = torch.arange(bt_size)[:, None].to(pred_idx_pad.device).repeat(1, pt_num).view(-1, )
    p_ids = pred_idx_pad.view(-1, )
    p_ids_ft = idx * pt_num + p_ids
    p_ids_ft = torch.where(p_ids < 0, torch.zeros_like(p_ids_ft), p_ids_ft)
    pred_poly_pad = py_pred.view(-1, 2)[p_ids_ft]
    pred_poly_pad = pred_poly_pad.view(bt_size, pt_num, 2)
    pred_poly_pad = \
        torch.where(pred_idx_pad[..., None].repeat(1, 1, 2) < 0, torch.zeros_like(pred_poly_pad), pred_poly_pad)
    if b_ids.size(0) > 0:
        k_ids = (keep_num[:, 0] != 0).nonzero()[:, 0]
        gt_poly_pad = gt_poly_pad[k_ids]
        gt_idx_pad = gt_idx_pad[k_ids]
        pred_idx_pad = pred_idx_pad[k_ids]
        pred_poly_pad = pred_poly_pad[k_ids]
        bt_size = k_ids.size(0)

    pred_match = compute_match(pred_idx_pad, pred_poly_pad, gt_idx_pad, gt_poly_pad)
    n_pred_idx_pad, it_pys, pred_match = random_noise(pred_idx_pad, pred_poly_pad, pred_match, iter_i)

    up_it_pys, start_idx, edge_num = uniform_upsample(it_pys, n_pred_idx_pad, min_v=4)
    up_it_pys = up_it_pys[0]
    start_idx = start_idx[0]
    k_idx = (start_idx < pt_num).nonzero()
    start_idx = start_idx.view(-1,)[k_idx[:, 0] * pt_num + k_idx[:, 1]]
    p_02 = torch.zeros(bt_size * pt_num).to(cls.device)
    p_02[k_idx[:, 0] * pt_num + start_idx] = 2
    p_02 = p_02.view(bt_size, pt_num)

    s_ids = pred_match
    up_gt_pys, start_idx = upsample_targets(gt_poly_pad, s_ids, edge_num[0], gt_idx_pad)
    up_gt_pys = up_gt_pys[0]
    start_idx = start_idx[0]
    s1 = start_idx.size(1)
    k_idx = (start_idx < pt_num).nonzero()
    start_idx = start_idx.view(-1,)[k_idx[:, 0] * s1 + k_idx[:, 1]]
    p_02 = p_02.view(-1,)
    p_02_v = p_02[k_idx[:, 0] * pt_num + start_idx]
    p_02[k_idx[:, 0] * pt_num + start_idx] = torch.max(p_02_v, torch.full_like(p_02_v, 1))
    p_02 = p_02.view(bt_size, pt_num)
    p_iter_cls = get_iter_cls(up_gt_pys, p_02)

    if b_ids.size(0) > 0:
        i_iter_it_pys = torch.zeros_like(i_gt_py)
        i_iter_gt_pys = torch.zeros_like(i_gt_py)
        iter_01 = torch.zeros_like(gt_cls)
        iter_cls = torch.zeros_like(gt_cls[:, :, None])
        i_iter_it_pys[k_ids] = up_it_pys
        i_iter_gt_pys[k_ids] = up_gt_pys
        iter_01[k_ids] = p_02
        iter_cls[k_ids] = p_iter_cls
    else:
        i_iter_it_pys = up_it_pys
        i_iter_gt_pys = up_gt_pys
        iter_01 = p_02
        iter_cls = p_iter_cls

    c_iter_it_pys = snake_gcn_utils.img_poly_to_can_poly(i_iter_it_pys)
    c_iter_gt_pys = snake_gcn_utils.img_poly_to_can_poly(i_iter_gt_pys)

    iter_init = {}
    iter_init.update({'i_iter_it_py': i_iter_it_pys.to(i_gt_py.device)})
    iter_init.update({'i_iter_it_cls': (iter_01 == 2).float().to(i_gt_py.device)})
    iter_init.update({'c_iter_it_py': c_iter_it_pys.to(i_gt_py.device)})
    iter_init.update({'i_iter_gt_py': i_iter_gt_pys.to(i_gt_py.device)})
    iter_init.update({'c_iter_gt_py': c_iter_gt_pys.to(i_gt_py.device)})
    iter_init.update({'iter_gt_cls': iter_cls.to(i_gt_py.device)})
    iter_init.update({'iter_01': iter_01.to(i_gt_py.device)})

    return iter_init


def prepare_testing_iter(py_pred, cls):
    """
    :param py_pred: (b, n, 2)
    :param cls: (b, n, 1)
    :return:
    """
    assert py_pred.size(0) == cls.size(0)

    bt_size = py_pred.size(0)
    pt_num = py_pred.size(1)

    cls = cls[..., 0]

    # prepare pred_idx_pad
    pred_idx = (cls > cfg.s).nonzero()
    pred_num = torch.sum((cls > cfg.s).int(), 1)[:, None].repeat(1, pt_num).view(-1, )
    arange_ids = torch.arange(bt_size * pt_num).to(py_pred.device) % pt_num
    # pred_idx_pad (b, n) pad -1
    pred_idx_pad = torch.full_like(arange_ids, -1)
    idx = (arange_ids < pred_num).nonzero()[:, 0]
    pred_idx_pad[idx] = pred_idx[:, 1]
    pred_idx_pad = pred_idx_pad.reshape(bt_size, pt_num)
    pred_idx_pad = add_pred_cls(pred_idx_pad)
    # pred_poly_pad (b, n, 2) pad 0
    b_ids = torch.arange(bt_size)[:, None].to(pred_idx_pad.device).repeat(1, pt_num).view(-1,)
    p_ids = pred_idx_pad.view(-1,)
    p_ids_ft = b_ids * pt_num + p_ids
    p_ids_ft = torch.where(p_ids < 0, torch.zeros_like(p_ids_ft), p_ids_ft)
    pred_poly_pad = py_pred.view(-1, 2)[p_ids_ft]
    pred_poly_pad = pred_poly_pad.view(bt_size, pt_num, 2)
    pred_poly_pad = \
        torch.where(pred_idx_pad[..., None].repeat(1, 1, 2) < 0, torch.zeros_like(pred_poly_pad), pred_poly_pad)
    up_it_pys, start_idx, edge_num = uniform_upsample(pred_poly_pad, pred_idx_pad, min_v=4)
    i_iter_it_pys = up_it_pys[0]
    start_idx = start_idx[0]
    k_idx = (start_idx < pt_num).nonzero()
    start_idx = start_idx.view(-1, )[k_idx[:, 0] * pt_num + k_idx[:, 1]]
    p_01 = torch.zeros(bt_size * pt_num).to(cls.device)
    p_01[k_idx[:, 0] * pt_num + start_idx] = 2
    iter_01 = p_01.view(bt_size, pt_num)

    c_iter_it_pys = snake_gcn_utils.img_poly_to_can_poly(i_iter_it_pys)

    iter_init = {}
    iter_init.update({'i_iter_it_py': i_iter_it_pys.to(py_pred.device)})
    iter_init.update({'i_iter_it_cls': (iter_01 == 2).float().to(py_pred.device)})
    iter_init.update({'c_iter_it_py': c_iter_it_pys.to(py_pred.device)})
    iter_init.update({'iter_01': iter_01.to(py_pred.device)})

    return iter_init


def get_cls_01(iter_01, radius):
    """
    :param iter_01: (b, n)
    :param radius:
    :return:
    """
    cls_2 = (iter_01 == 2).float()
    r_cls_2 = torch.cat([torch.roll(cls_2, i, 1)[..., None] for i in range(-radius, radius + 1)], 2)
    if r_cls_2.size(0) > 0:
        r_cls_2, _ = torch.max(r_cls_2, dim=2)
        cls_01 = r_cls_2
    else:
        cls_01 = cls_2
    cls_01 = 1 - cls_01

    return cls_01


def _dwt_dist(poly1, poly2, s_id1, s_id2, m_d):
    """
    :param poly1: (n, 2)
    :param poly2: (m, 2)
    :param m_d: (n, m)
    :return:
    """
    o_id1 = np.arange(poly1.shape[0])
    o_id1 = np.concatenate([o_id1[s_id1:], o_id1[:s_id1 + 1]], axis=0)
    o_id2 = np.arange(poly2.shape[0])
    o_id2 = np.concatenate([o_id2[s_id2:], o_id2[:s_id2 + 1]], axis=0)
    poly1 = poly1[o_id1]
    poly2 = poly2[o_id2]
    n, m = poly1.shape[0], poly2.shape[0]

    cost = np.ones((n, m))
    cost[0, 0] = m_d[o_id1[0], o_id2[0]]
    for i in range(1, n):
        cost[i, 0] = cost[i - 1, 0] + m_d[o_id1[i], o_id2[0]]
    for j in range(1, m):
        cost[0, j] = cost[0, j - 1] + m_d[o_id1[0], o_id2[j]]
    for i in range(1, n):
        for j in range(1, m):
            cost[i, j] = m_d[o_id1[i], o_id2[j]] + min(cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j])

    match = np.zeros((n - 1, m - 1))
    i, j = n - 1, m - 1
    match[o_id1[i], o_id2[j]] = 1
    while (i > 0) or (j > 0):
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            tb = np.argmin((cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]))
            if tb == 0:
                i -= 1
                j -= 1
            elif tb == 1:
                i -= 1
            else:
                j -= 1
        match[o_id1[i], o_id2[j]] = 1
    return match


def compute_match(idx1, poly1, idx2, poly2):
    """
    :param idx1: (b, n) -1 pad.
    :param poly1: (b, n, 2)
    :param idx2: (b, m) -1 pad.
    :param poly2: (b, m, 2)
    :return: pred_match: (b, n) -1 pad.
    """
    assert idx1.size(0) == idx2.size(0) == poly1.size(0) == poly2.size(0)
    idx1_tensor = idx1.clone()
    idx1 = idx1.detach().cpu().numpy()
    poly1 = poly1.detach().cpu().numpy()
    idx2 = idx2.detach().cpu().numpy()
    poly2 = poly2.detach().cpu().numpy()

    b = idx1.shape[0]
    pt_num = idx1.shape[1]
    pred_match = np.full((b, pt_num), -1)

    for i in range(b):
        idx1_i = idx1[i]
        idx2_i = idx2[i]
        poly1_i = poly1[i]
        poly2_i = poly2[i]
        poly1_i = poly1_i[np.where(idx1_i >= 0)]
        poly2_i = poly2_i[np.where(idx2_i >= 0)]
        p_num1 = poly1_i.shape[0]
        p_num2 = poly2_i.shape[0]
        m_poly1_i = np.tile(poly1_i[:, None], (1, p_num2, 1))
        m_poly2_i = np.tile(poly2_i[None], (p_num1, 1, 1))
        m_d_i = np.sqrt(np.sum(np.power((m_poly1_i - m_poly2_i), 2), axis=2))
        min_v = np.min(m_d_i)
        id1, id2 = np.where(m_d_i == min_v)
        s_id1, s_id2 = id1[0], id2[0]
        dwt_match = _dwt_dist(poly1_i, poly2_i, s_id1, s_id2, m_d_i)

        # compute significance of target points
        pre_p2 = np.roll(poly2_i, 1, axis=0)
        post_p2 = np.roll(poly2_i, -1, axis=0)
        pre_v = pre_p2 - poly2_i
        post_v = post_p2 - poly2_i
        w_p2 = np.abs(pre_v[:, 0] * post_v[:, 1] - pre_v[:, 1] * post_v[:, 0])

        p_sum = np.sum(dwt_match, axis=1)
        multi_pid = np.asarray(np.where(p_sum > 1)[0])
        for j in multi_pid:
            t_ids = np.asarray(np.where(dwt_match[j] == 1)[0])
            t_sum = dwt_match.sum(0)
            ind1 = np.asarray(np.where(t_sum[t_ids] == 1)[0])
            ind2 = np.asarray(np.where(t_sum[t_ids] > 1)[0])
            if ind2.shape[0] > 0:
                dwt_match[j, t_ids[ind2]] = 0
            if ind1.shape[0] == 0:
                min_id = np.argmin(dwt_match[j, t_ids[ind2]])
                dwt_match[j, t_ids[ind2[min_id]]] = 1
            if ind1.shape[0] > 0:
                dwt_match[j, t_ids[ind1]] = 0
                sele_id = np.argmax(w_p2[t_ids[ind1]])
                dwt_match[j, t_ids[ind1[sele_id]]] = 1
        assert np.all(np.sum(dwt_match, axis=1) == 1)
        pred_match[i, np.where(idx1_i >= 0)[0]] = np.argmax(dwt_match, axis=1)
    pred_match = torch.tensor(pred_match).type_as(idx1_tensor).to(idx1_tensor.device)
    return pred_match

##########################################################################
# loss utils
##########################################################################
def _smooth_l1_loss(pred, targets, weights):
    """
    :param pred: [b, n, 2]
    :param targets: [b, n, 2]
    :param weights: [b, n, 1]
    """
    diff = torch.abs(pred - targets)
    in_loss = torch.where(diff < 1, torch.pow(diff, 2), diff - 0.5)
    weights = weights.repeat(1, 1, in_loss.size(2))
    in_loss = in_loss * weights
    in_loss = torch.sum(in_loss) / (torch.sum(weights) + 1e-3)
    return in_loss


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()
        self.smooth_l1_loss = _smooth_l1_loss

    def forward(self, preds, targets, weights):
        return self.smooth_l1_loss(preds, targets, weights)


def _neg_loss(pred, gt, weights=None):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
            pred (b, n, 1)
            gt (b, n, 1)
            weights (b, n, 1)
    '''
    if weights is None:
        weights = torch.full_like(pred, 1)

    pos_inds = gt.eq(1).float() * weights
    neg_inds = gt.lt(1).float() * weights

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds


    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target, weights=None):
        return self.neg_loss(out, target, weights)