import torch.nn as nn
from .snake import Snake
from lib.utils.snake import snake_gcn_utils, snake_config
import torch
from lib.config import cfg
from lib.utils import poly_utils

class Evolution(nn.Module):
    def __init__(self):
        super(Evolution, self).__init__()

        self.fuse = nn.Conv1d(128, 64, 1)
        self.init_gcn = Snake(state_dim=128, feature_dim=64 + 2, conv_type='dgrid')
        self.evolve_gcn = Snake(state_dim=128, feature_dim=64 + 2, conv_type='dgrid', is_cls=True)
        self.iter = cfg.iter
        for i in range(self.iter):
            evolve_gcn = Snake(state_dim=128, feature_dim=64 + 2 + 1, conv_type='dgrid', is_cls=True)
            self.__setattr__('evolve_gcn'+str(i), evolve_gcn)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def prepare_training(self, output, batch):
        init = snake_gcn_utils.prepare_training(output, batch)
        output.update({'i_it_4py': init['i_it_4py'], 'i_it_py': init['i_it_py']})
        output.update({'i_gt_4py': init['i_gt_4py'], 'i_gt_py': init['i_gt_py']})
        output.update({'cls_gt_py': init['cls_gt_py']})
        return init

    def prepare_training_iter(self, py_pred, cls, gt_cls, i_gt_py, iter_i):
        iter_init = poly_utils.prepare_training_iter(py_pred, cls, gt_cls, i_gt_py, iter_i)
        return iter_init

    def prepare_testing_init(self, output):
        init = snake_gcn_utils.prepare_testing_init(output['detection'][..., :4], output['detection'][..., 4])
        output['detection'] = output['detection'][output['detection'][..., 4] > snake_config.ct_score]
        output.update({'it_ex': init['i_it_4py']})
        return init

    def prepare_testing_iter(self, py_pred, cls):
        iter_init = poly_utils.prepare_testing_iter(py_pred, cls)
        return iter_init

    def prepare_testing_evolve(self, output, h, w):
        ex = output['ex']
        ex[..., 0] = torch.clamp(ex[..., 0], min=0, max=w-1)
        ex[..., 1] = torch.clamp(ex[..., 1], min=0, max=h-1)
        evolve = snake_gcn_utils.prepare_testing_evolve(ex)
        output.update({'it_py': evolve['i_it_py']})
        return evolve

    def init_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind):
        if len(i_it_poly) == 0:
            return torch.zeros([0, 4, 2]).to(i_it_poly)

        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)
        center = (torch.min(i_it_poly, dim=1)[0] + torch.max(i_it_poly, dim=1)[0]) * 0.5
        ct_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, center[:, None], ind, h, w)
        init_feature = torch.cat([init_feature, ct_feature.expand_as(init_feature)], dim=1)
        init_feature = self.fuse(init_feature)

        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, init_input.size(2), init_input.device)
        i_poly = i_it_poly + snake(init_input, adj).permute(0, 2, 1)
        i_poly = i_poly[:, ::snake_config.init_poly_num//4]

        return i_poly

    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind, it_cls=None):
        if len(i_it_poly) == 0:
            t1 = torch.zeros_like(i_it_poly)
            if snake.cls is not None:
                t2 = torch.zeros_like(i_it_poly)[:, :, :1]
                return t1, t2
            return t1
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)
        c_it_poly = c_it_poly * snake_config.ro
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        if it_cls is not None:
            init_input = torch.cat([init_input, it_cls[:, None]], dim=1)
        adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, init_input.size(2), init_input.device)
        if snake.cls is not None:
            x, cls = snake(init_input, adj)
            i_poly = i_it_poly * snake_config.ro + x.permute(0, 2, 1)
            return i_poly, cls.permute(0, 2, 1)
        else:
            i_poly = i_it_poly * snake_config.ro + snake(init_input, adj).permute(0, 2, 1)
        return i_poly

    def forward(self, output, cnn_feature, batch=None):
        ret = output

        if batch is not None and 'test' not in batch['meta']:
            with torch.no_grad():
                init = self.prepare_training(output, batch)

            ex_pred = self.init_poly(self.init_gcn, cnn_feature, init['i_it_4py'], init['c_it_4py'], init['4py_ind'])
            ret.update({'ex_pred': ex_pred, 'i_gt_4py': output['i_gt_4py']})

            py_pred, py_cls = self.evolve_poly(
                self.evolve_gcn, cnn_feature, init['i_it_py'], init['c_it_py'], init['py_ind'])
            nms_cls = poly_utils.cls_nms(torch.sigmoid(py_cls))
            py_preds = [py_pred]
            py_gts = [output['i_gt_py'] * snake_config.ro]
            py_clses = [py_cls]
            nms_clses = [nms_cls]
            gt_clses = [output['cls_gt_py']]
            iter_01s = []
            iter_init_pys = []
            for i in range(self.iter):
                if py_pred.size(0) == 0:
                    break
                py_pred = py_pred / snake_config.ro
                if i == 0:
                    cls = nms_cls
                    gt_cls = output['cls_gt_py']
                    i_gt_py = output['i_gt_py']
                else:
                    cls = nms_cls
                    gt_cls = iter_init['iter_01'][..., None]
                    i_gt_py = iter_init['i_iter_gt_py']
                iter_init = self.prepare_training_iter(py_pred, cls, gt_cls, i_gt_py, i)
                evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
                py_pred, py_cls = \
                    self.evolve_poly(evolve_gcn, cnn_feature, iter_init['i_iter_it_py'], iter_init['c_iter_it_py'],
                                     init['py_ind'], it_cls=iter_init['i_iter_it_cls'])
                nms_cls = poly_utils.cls_nms(
                    torch.sigmoid(py_cls), radius=2, p_01=iter_init['iter_01'], py_pred=None)
                py_preds.append(py_pred)
                py_gts.append(iter_init['i_iter_gt_py'] * snake_config.ro)
                py_clses.append(py_cls)
                nms_clses.append(nms_cls)
                gt_clses.append(iter_init['iter_gt_cls'])
                iter_01s.append(iter_init['iter_01'])
                iter_init_pys.append(iter_init['i_iter_it_py'] * snake_config.ro)
            ret.update({'py_pred': py_preds, 'py_gt': py_gts})
            ret.update({'py_cls': py_clses, 'nms_cls': nms_clses, 'gt_cls': gt_clses, 'iter_01': iter_01s})
            ret.update({'iter_init_py': iter_init_pys})


        if not self.training:
            with torch.no_grad():
                init = self.prepare_testing_init(output)
                ex = self.init_poly(self.init_gcn, cnn_feature, init['i_it_4py'], init['c_it_4py'], init['ind'])
                ret.update({'ex': ex})
                evolve = self.prepare_testing_evolve(output, cnn_feature.size(2), cnn_feature.size(3))

                py, cls = self.evolve_poly(
                    self.evolve_gcn, cnn_feature, evolve['i_it_py'], evolve['c_it_py'], init['ind'])
                nms_cls = poly_utils.cls_nms(torch.sigmoid(cls))
                pys = [py / snake_config.ro]
                clses = [cls]
                nms_clses = [nms_cls]
                it_clses = []
                it_pys = [evolve['i_it_py']]
                for i in range(self.iter):
                    if py.size(0) == 0:
                        break
                    py = py / snake_config.ro
                    iter_init = self.prepare_testing_iter(py, nms_cls)
                    evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
                    py, cls = \
                        self.evolve_poly(evolve_gcn, cnn_feature, iter_init['i_iter_it_py'], iter_init['c_iter_it_py'],
                                         init['ind'], it_cls=iter_init['i_iter_it_cls'])
                    nms_cls = poly_utils.cls_nms(torch.sigmoid(cls), radius=2,
                                                 p_01=iter_init['iter_01'], py_pred=py)

                    pys.append(py / snake_config.ro)
                    clses.append(torch.sigmoid(cls))
                    nms_clses.append(nms_cls)
                    it_clses.append(iter_init['i_iter_it_cls'])
                    it_pys.append(iter_init['i_iter_it_py'])
                ret.update({'py': pys, 'cls': clses, 'nms_cls': nms_clses, 'it_cls': it_clses,
                            'it_pys': it_pys})

        return output

