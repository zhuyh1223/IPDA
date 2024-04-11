import torch.nn as nn
from lib.utils import net_utils, poly_utils
from lib.config import cfg
import torch


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net

        self.ct_crit = net_utils.FocalLoss()
        self.wh_crit = net_utils.IndL1Loss1d('smooth_l1')
        self.reg_crit = net_utils.IndL1Loss1d('smooth_l1')
        self.ex_crit = torch.nn.functional.smooth_l1_loss
        self.py_crit = torch.nn.functional.smooth_l1_loss
        self.py_crit_iter = poly_utils.SmoothL1Loss()
        self.cls_crit = poly_utils.FocalLoss()

    def forward(self, batch, is_cls=True):
        output = self.net(batch['inp'], batch)

        scalar_stats = {}
        loss = 0

        ct_loss = self.ct_crit(net_utils.sigmoid(output['ct_hm']), batch['ct_hm'])
        scalar_stats.update({'ct_loss': ct_loss})
        loss += ct_loss

        wh_loss = self.wh_crit(output['wh'], batch['wh'], batch['ct_ind'], batch['ct_01'])
        scalar_stats.update({'wh_loss': wh_loss})
        loss += 0.1 * wh_loss

        ex_loss = self.ex_crit(output['ex_pred'], output['i_gt_4py'])
        scalar_stats.update({'ex_loss': ex_loss})
        loss += ex_loss

        py_loss = 0
        cls_loss = 0

        iter_01 = output['gt_cls'][0][..., 0]
        weight = torch.where(iter_01 > 0, torch.full_like(iter_01, 1.0), torch.full_like(iter_01, 0.1))
        py_loss += self.py_crit_iter(output['py_pred'][0], output['py_gt'][0], weight[..., None]) / len(
            output['py_pred'])
        cls_loss += self.cls_crit(net_utils.sigmoid(output['py_cls'][0]), output['gt_cls'][0]) / len(output['py_cls'])
        for i in range(len(output['py_pred']) - 1):
            iter_01 = output['iter_01'][i]
            weight = 1 - poly_utils.get_cls_01(2 * (iter_01 > 0).type_as(iter_01), 2)
            weight = torch.where(iter_01 > 0, torch.full_like(weight, 1.0), torch.full_like(weight, 0.1))
            py_loss += self.py_crit_iter(output['py_pred'][i + 1], output['py_gt'][i + 1], weight[..., None]) / len(
                output['py_pred'])
            cls_01 = poly_utils.get_cls_01(iter_01, 2)
            cls_loss += self.cls_crit(net_utils.sigmoid(output['py_cls'][i + 1]), output['gt_cls'][i + 1],
                                      cls_01[..., None]) / len(output['py_cls'])

        scalar_stats.update({'py_loss': py_loss})
        loss += py_loss

        if is_cls:
            scalar_stats.update({'cls_loss': cls_loss})
            loss += cls_loss


        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats

