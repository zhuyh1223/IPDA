import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

from lib.config import cfg, args
from lib.datasets import make_data_loader
from lib.networks import make_network
from lib.utils.net_utils import load_network
from lib.visualizers import make_visualizer
from lib.utils import metric_pro, img_utils
from lib.utils.snake import snake_config

network = make_network(cfg).cuda()
model_dir = cfg.model_dir
load_network(network, model_dir, resume=cfg.resume, epoch=cfg.test.epoch, strict=False)
network.eval()
val_loader = make_data_loader(cfg, is_train=False, split='test')

def get_gts(batch):
    png_path = batch['meta']['png_path'][0]
    txt_path = batch['meta']['txt_path'][0]
    original_image = img_utils.bgr_to_rgb(cv2.imread(png_path))
    polys, masks = metric_pro.generate_gts_from_txt(original_image, txt_path)
    return original_image, polys, masks

def get_preds(image, inp, output):
    height, width = image.shape[:2]
    inp = inp.detach().cpu().numpy()
    inp_h, inp_w = inp.shape[2:]
    if len(output['py']) == 0:
        return np.zeros((1, 1, 2)), np.zeros((height, width, 1)), np.zeros(1)
    py_out = output['py'][-1]
    cls = output['nms_cls'][-1][..., 0]
    py_pred = py_out.cpu().numpy() * snake_config.down_ratio
    py_pred = np.round(py_pred * [width / inp_w, height / inp_h])
    cls = cls.cpu().numpy()

    boxes = output['detection'][:, :4].detach().cpu().numpy() * snake_config.down_ratio
    boxes = np.round(boxes * [width / inp_w, height / inp_h, width / inp_w, height / inp_h])
    scores = output['detection'][:, 4].detach().cpu().numpy()
    assert scores.shape[0] == len(py_pred)

    if len(py_pred) == 0:
        return np.zeros((1, 1, 2)), np.zeros((height, width, 1)), np.zeros(1)

    polys = []
    masks = []

    for i in range(len(py_pred)):
        poly = np.zeros((1, 2))
        mask = np.zeros((height, width))
        idx = np.where(cls[i] > cfg.s)
        if len(idx[0]) > 0:
            poly = py_pred[i][idx]
            cv2.drawContours(mask, [np.expand_dims(poly, axis=1).astype(np.int32)], 0, 255, cv2.FILLED)
        polys.append(poly)
        masks.append(mask)
    masks = np.stack(masks, axis=2)
    masks = (masks > 0).astype(np.int32)
    return polys, masks, scores

def to_cuda(batch):
    for k in batch:
        if isinstance(batch[k], dict):
            to_cuda(batch[k])
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].cuda()
    return batch


def draw(res_dir):
    visualizer = make_visualizer(cfg)
    for i, batch in enumerate(tqdm(val_loader)):

        original_image, gt_polys, gt_masks = get_gts(batch)
        batch = to_cuda(batch)
        with torch.no_grad():
            output = network(batch['inp'], batch)

        alpha_c = np.expand_dims(200 * np.ones(np.shape(original_image)[:2]), axis=-1)
        masked_image = np.concatenate([original_image, alpha_c], axis=-1)
        masked_image = masked_image.astype(np.uint8)

        polys, pred_masks, _ = get_preds(original_image, batch['inp'], output)
        visualizer.visualize_gts(masked_image, gt_polys,
                                 save_path="%s%d_gt.png" % (res_dir, i))
        visualizer.visualize_preds(masked_image, polys,
                                   save_path="%s%d_pred.png" % (res_dir, i))


def metric(metric_path):
    ap_array = np.zeros(len(val_loader))
    ap50_array = np.zeros(len(val_loader))
    ap75_array = np.zeros(len(val_loader))
    poly_sim_array = np.zeros(len(val_loader))
    gt_size_array = np.zeros(len(val_loader))
    count = 0
    i = 0
    for i, batch in enumerate(tqdm(val_loader)):
        original_image, gt_polys, gt_masks = get_gts(batch)
        batch = to_cuda(batch)
        with torch.no_grad():
            output = network(batch['inp'], batch)
        pred_polys, pred_masks, pred_scores = get_preds(original_image, batch['inp'], output)

        if np.sum(pred_masks) == 0:
            ap = round(0.0, 6)
            ap50 = round(0.0, 6)
            ap75 = round(0.0, 6)
            poly_sim = round(0.0, 6)
            gt_size = round(float(np.sum(gt_masks)), 6)
        else:
            ap, ap50, ap75, poly_sim, gt_size = \
                metric_pro.compute_metrics(gt_masks, pred_scores, pred_masks, gt_polys, pred_polys)

        ap_array[i] = ap
        ap50_array[i] = ap50
        ap75_array[i] = ap75
        poly_sim_array[i] = poly_sim
        gt_size_array[i] = gt_size
        print("image_id:%d  ap:%f, ap50:%f, ap75:%f, poly_sim:%f, gt_size:%f" % (i, ap, ap50, ap75, poly_sim, gt_size))

        with open(metric_path, "w" if i == 0 else "a") as f:
            f.write(
                " ".join(map(str, [ap, ap50, ap75, poly_sim, gt_size])))
            f.write("\n")
        count += 1

    ap_array = ap_array[:count]
    ap50_array = ap50_array[:count]
    ap75_array = ap75_array[:count]
    poly_sim_array = poly_sim_array[:count]
    gt_size_array = gt_size_array[:count]

    ap = np.mean(ap_array)
    ap50 = np.mean(ap50_array)
    ap75 = np.mean(ap75_array)
    poly_sim = float(np.sum(poly_sim_array)) / float(np.sum(gt_size_array))

    print("RESULT-->> ap:%f, ap50:%f, ap75:%f, poly_sim:%f" % (ap, ap50, ap75, poly_sim))

    with open(metric_path, "w" if i == 0 else "a") as f:
        f.write(" ".join(map(str, [ap, ap50, ap75, poly_sim])))
        f.write("\n")


def speed(save_path):
    import time
    from thop import profile

    mode = "w" if not os.path.exists(save_path) else "a"
    total_time = 0
    for i, batch in enumerate(tqdm(val_loader)):
        batch['inp'] = torch.FloatTensor(batch['inp']).cuda()
        with torch.no_grad():
            if i == 0:
                flops, params = profile(network, inputs=(batch['inp'], batch))
                with open(save_path, mode) as f:
                    f.write("IPDA:{}\n".format(cfg.model))
                    f.write("flops: %.2f M, params: %.2f M\n" % (flops / 1000000.0, params / 1000000.0))
                print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
            torch.cuda.synchronize()
            start = time.time()
            network(batch['inp'], batch)
            torch.cuda.synchronize()
            total_time += time.time() - start
    with open(save_path, "a") as f:
        f.write("time: %fms\n" % (total_time / len(val_loader) * 1000))
        print("time: %fms" % (total_time / len(val_loader) * 1000))


if __name__ == '__main__':

    assert args.type in ['visualize', 'evaluate', 'speed']

    if args.type == 'visualize':
        res_dir = 'output/{}/show/'.format(cfg.model)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        draw(res_dir)
    elif args.type == 'evaluate':
        res_dir = 'output/{}/'.format(cfg.model)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        metric_path = 'output/{}/metric.txt'.format(cfg.model)
        metric(metric_path)
    elif args.type == 'speed':
        save_path = 'output/{}/time.txt'.format(cfg.model)
        speed(save_path)
