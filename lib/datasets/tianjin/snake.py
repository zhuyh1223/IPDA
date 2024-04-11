import os
from lib.utils.snake import snake_voc_utils, snake_config
import cv2
import numpy as np
import math
from lib.utils import data_utils, poly_utils
import torch.utils.data as data
from lib.config import cfg
import json


class Dataset(data.Dataset):
    def __init__(self, png_dir, txt_dir, split):
        super(Dataset, self).__init__()

        self.png_dir = png_dir
        self.txt_dir = txt_dir
        self.split = split
        self.images = []

        png_list = os.listdir(self.png_dir)
        png_list.sort()
        count = 0
        for i in png_list:
            file_name = os.path.splitext(i)[0]
            png_path = os.path.join(self.png_dir, i)
            txt_path = os.path.join(self.txt_dir, "%s.txt" % file_name)
            self.images.append({
                "image_id": count,
                "file_name": file_name,
                "png_path": png_path,
                "txt_path": txt_path
            })
            count += 1

    def generate_polys_from_txt(self, txt_path):
        polys = []
        with open(txt_path, 'r') as f:
            data = f.readlines()
            for j, line in enumerate(data):
                odom = line.split()
                odom = np.array(list(map(int, odom)))
                cnt = np.reshape(odom, (int(odom.shape[0] / 2), 1, 2))

                epsilon = 1
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                polys.append(np.expand_dims(approx[:, 0], axis=0))
        return polys

    def read_original_data(self, png_path, txt_path):
        img = cv2.imread(png_path)
        instance_polys = self.generate_polys_from_txt(txt_path)
        return img, instance_polys

    def transform_original_data(self, instance_polys, flipped, width, trans_output, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            polys = [poly.reshape(-1, 2) for poly in instance]

            if flipped:
                polys_ = []
                for poly in polys:
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1
                    polys_.append(poly.copy())
                polys = polys_

            polys = snake_voc_utils.transform_polys(polys, trans_output, output_h, output_w)
            instance_polys_.append(polys)
        return instance_polys_

    def get_valid_polys(self, instance_polys, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 3]
            for poly in instance:
                poly[:, 0] = np.clip(poly[:, 0], 0, output_w - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, output_h - 1)
            polys = snake_voc_utils.filter_tiny_polys(instance)
            polys = snake_voc_utils.get_cw_polys(polys)
            polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
            instance_polys_.append(polys)
        return instance_polys_

    def get_extreme_points(self, instance_polys):
        extreme_points = []
        for instance in instance_polys:
            points = [snake_voc_utils.get_extreme_points(poly) for poly in instance]
            extreme_points.append(points)
        return extreme_points

    def prepare_detection(self, box, poly, ct_hm, wh, ct_cls, ct_ind):
        ct_hm = ct_hm[0]
        ct_cls.append(0)

        x_min, y_min, x_max, y_max = box
        ct = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)
        ct = np.round(ct).astype(np.int32)

        h, w = y_max - y_min, x_max - x_min
        radius = data_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        data_utils.draw_umich_gaussian(ct_hm, ct, radius)

        wh.append([w, h])
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])

        x_min, y_min = ct[0] - w / 2, ct[1] - h / 2
        x_max, y_max = ct[0] + w / 2, ct[1] + h / 2
        decode_box = [x_min, y_min, x_max, y_max]

        return decode_box

    def prepare_detection_(self, box, poly, ct_hm, cls_id, wh, ct_cls, ct_ind):
        ct_hm = ct_hm[cls_id]
        ct_cls.append(cls_id)

        x_min, y_min, x_max, y_max = box
        box_ct = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)

        x_min_int, y_min_int = int(x_min), int(y_min)
        h_int, w_int = math.ceil(y_max - y_min_int) + 1, math.ceil(x_max - x_min_int) + 1
        max_h, max_w = ct_hm.shape[0], ct_hm.shape[1]
        h_int, w_int = min(y_min_int + h_int, max_h) - y_min_int, min(x_min_int + w_int, max_w) - x_min_int

        mask_poly = poly - np.array([x_min_int, y_min_int])
        mask_ct = box_ct - np.array([x_min_int, y_min_int])
        ct, off, xy = snake_voc_utils.prepare_ct_off_mask(mask_poly, mask_ct, h_int, w_int)

        xy += np.array([x_min_int, y_min_int])
        ct += np.array([x_min_int, y_min_int])

        h, w = y_max - y_min, x_max - x_min
        radius = data_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        data_utils.draw_umich_gaussian(ct_hm, ct, radius)

        wh.append([w, h])
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])

    def prepare_init(self, box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, h, w):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        img_init_poly = snake_voc_utils.get_init(box)
        img_init_poly = snake_voc_utils.uniformsample(img_init_poly, snake_config.init_poly_num)
        can_init_poly = snake_voc_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)
        img_gt_poly = extreme_point
        can_gt_poly = snake_voc_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        i_it_4pys.append(img_init_poly)
        c_it_4pys.append(can_init_poly)
        i_gt_4pys.append(img_gt_poly)
        c_gt_4pys.append(can_gt_poly)

    def prepare_evolution(self, poly, extreme_point, img_init_polys, can_init_polys, img_gt_polys, can_gt_polys,
                          cls_gt_polys, i_gt_points, pt_num):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        octagon = snake_voc_utils.get_octagon(extreme_point)
        img_init_poly = snake_voc_utils.uniformsample(octagon, snake_config.poly_num)
        can_init_poly = snake_voc_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)

        img_gt_poly = snake_voc_utils.uniformsample(poly, snake_config.gt_poly_num)

        area_init = cv2.contourArea(np.round(img_init_poly[:, None]).astype(np.int32), True)
        area_gt = cv2.contourArea(np.round(img_gt_poly[:, None]).astype(np.int32), True)
        if area_init * area_gt < 0:
            img_gt_poly = img_gt_poly[::-1]

        tt_idx = np.argmin(np.power(img_gt_poly - img_init_poly[0], 2).sum(axis=1))
        img_gt_poly = np.roll(img_gt_poly, -tt_idx, axis=0)
        can_gt_poly = snake_voc_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        cls_gt_poly = poly_utils.get_corner_cls2(img_gt_poly, poly)

        img_init_polys.append(img_init_poly)
        can_init_polys.append(can_init_poly)
        img_gt_polys.append(img_gt_poly)
        can_gt_polys.append(can_gt_poly)
        cls_gt_polys.append(cls_gt_poly)
        i_gt_points.append(poly)
        pt_num.append(poly.shape[0])

    def prepare_merge(self, is_id, cls_id, cp_id, cp_cls):
        cp_id.append(is_id)
        cp_cls.append(cls_id)

    def read_json(self, json_path, trans_output, inp_out_hw):
        with open(json_path, 'r') as f:
            ann = json.load(f)
        detections = np.zeros((100, 5))
        for i, instance in enumerate(ann):
            x1, y1, w, h = instance['bbox']
            detections[i, :4] = np.array([x1, y1, x1 + w, y1 + h])
            detections[i, 4] = instance['score']

        output_h, output_w = inp_out_hw[2:]
        poly = np.reshape(detections[:, :4], (-1, 2))
        poly = data_utils.affine_transform(poly, trans_output)
        poly[:, 0] = np.clip(poly[:, 0], 0, output_w - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, output_h - 1)
        detections[:, :4] = np.reshape(poly, (-1, 4))
        return detections

    def __getitem__(self, index):
        png_path = self.images[index]["png_path"]
        txt_path = self.images[index]["txt_path"]
        img_id = self.images[index]["image_id"]

        img, instance_polys = self.read_original_data(png_path, txt_path)


        height, width = img.shape[0], img.shape[1]
        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            snake_voc_utils.augment(
                img, self.split,
                snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
                snake_config.mean, snake_config.std, instance_polys
            )

        if self.split == 'test':
            ret = {'inp': inp}
            meta = {'center': center, 'scale': scale, 'img_id': img_id,
                    'png_path': png_path, 'txt_path': txt_path, 'test': ''}
            ret.update({'meta': meta})
            return ret

        instance_polys = self.transform_original_data(instance_polys, flipped, width, trans_output, inp_out_hw)
        instance_polys = self.get_valid_polys(instance_polys, inp_out_hw)
        extreme_points = self.get_extreme_points(instance_polys)

        # detection
        output_h, output_w = inp_out_hw[2:]
        ct_hm = np.zeros([cfg.heads.ct_hm, output_h, output_w], dtype=np.float32)
        wh = []
        ct_cls = []
        ct_ind = []

        # init
        i_it_4pys = []
        c_it_4pys = []
        i_gt_4pys = []
        c_gt_4pys = []

        # evolution
        i_it_pys = []
        c_it_pys = []
        i_gt_pys = []
        c_gt_pys = []
        cls_gt_pys = []
        i_gt_points = []
        pt_num = []

        for i in range(len(instance_polys)):
            instance_poly = instance_polys[i]
            instance_points = extreme_points[i]

            for j in range(len(instance_poly)):
                poly = instance_poly[j]
                extreme_point = instance_points[j]

                x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
                x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
                bbox = [x_min, y_min, x_max, y_max]
                h, w = y_max - y_min + 1, x_max - x_min + 1
                if h <= 1 or w <= 1:
                    continue

                self.prepare_detection(bbox, poly, ct_hm, wh, ct_cls, ct_ind)
                self.prepare_init(bbox, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, output_h, output_w)
                self.prepare_evolution(poly, extreme_point, i_it_pys, c_it_pys, i_gt_pys, c_gt_pys, cls_gt_pys,
                                       i_gt_points, pt_num)

        ret = {'inp': inp}
        detection = {'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind}
        init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
        evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys,
                     'cls_gt_py': cls_gt_pys}
        ret.update(detection)
        ret.update(init)
        ret.update(evolution)

        ct_num = len(ct_ind)
        meta = {'center': center, 'scale': scale, 'img_id': img_id, 'ann': img_id, 'ct_num': ct_num,
                'png_path': png_path, 'txt_path': txt_path
                }
        ret.update({'meta': meta})

        return ret

    def __len__(self):
        return len(self.images)

