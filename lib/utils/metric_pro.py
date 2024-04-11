import numpy as np
import cv2
import math

def generate_gts_from_txt(image, txt_path):
    mask_array = []
    polys = []
    height, width = image.shape[:2]
    with open(txt_path, 'r') as f:
        data = f.readlines()
        for j, line in enumerate(data):
            odom = line.split()
            odom = np.array(list(map(int, odom)))
            # cnt [points_num,1,(x,y)]
            cnt = np.reshape(odom, (int(odom.shape[0] / 2), 1, 2))
            mask = np.zeros((height, width))
            cv2.drawContours(mask, [cnt], 0, 255, cv2.FILLED)
            mask_array.append(mask)

            # epsilon = 0.005 * cv2.arcLength(cnt, True)
            epsilon = 1
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            polys.append(approx[:, 0])

        masks = np.stack(mask_array, axis=2)
        masks = (masks > 0).astype(np.int32)
    return polys, masks

def compute_overlaps_masks(masks1, masks2):
    '''Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    '''
    # flatten masks
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps

def compute_matches2(gt_masks, pred_masks, iou_threshold=0.0):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream

    # Compute IoU overlaps, precisions, recalls [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)
    flat_overlaps = np.reshape(overlaps.copy(), (-1,))
    pred_match = -1 * np.ones([pred_masks.shape[-1]])
    gt_match = -1 * np.ones([gt_masks.shape[-1]])
    for i in range(flat_overlaps.shape[0]):
        id_xy = np.argmax(flat_overlaps)
        score = flat_overlaps[id_xy]
        if score <= iou_threshold:
            break
        flat_overlaps[id_xy] = -1
        id_x = int(id_xy / overlaps.shape[1])
        id_y = int(id_xy % overlaps.shape[1])
        if pred_match[id_x] > -1 or gt_match[id_y] > -1:
            continue
        pred_match[id_x] = id_y
        gt_match[id_y] = id_x

    return gt_match, pred_match, overlaps

def compute_matches(gt_masks, pred_scores, pred_masks, iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_scores = pred_scores.copy()[indices]
    pred_masks = pred_masks.copy()[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks.copy())

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_masks.shape[-1]])
    gt_match = -1 * np.ones([gt_masks.shape[-1]])
    for i in range(pred_masks.shape[-1]):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > 0:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            match_count += 1
            gt_match[j] = i
            pred_match[i] = j
            break
    return gt_match, pred_match, overlaps

def compute_ap(gt_masks, pred_scores, pred_masks, iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_masks, pred_scores, pred_masks, iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps

def compute_ap_range(gt_mask, pred_score, pred_mask, iou_thresholds=None, verbose=0):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)

    # Compute AP over range of IoU thresholds
    AP = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps = \
            compute_ap(gt_mask, pred_score, pred_mask, iou_threshold=iou_threshold)
        if verbose:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
        AP.append(ap)
        if np.round(iou_threshold, 2) == 0.5:
            ap50 = ap
        if np.round(iou_threshold, 2) == 0.75:
            ap75 = ap
    AP = np.array(AP).mean()
    if verbose:
        print("AP @{:.2f}-{:.2f}:\t {:.3f}".format(
            iou_thresholds[0], iou_thresholds[-1], AP))
    return AP, ap50, ap75

def rank_polygon(poly):
    poly = poly.astype(np.int32)
    cnt = poly[:, None]
    area = cv2.contourArea(cnt, oriented=True)
    # rank clock-wise
    if area < 0:
        poly = poly[::-1]
        cnt = poly[:, None]
    # find the strat point
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cur_poly = poly - [cx, cy]
    post_poly = np.roll(cur_poly, -1, axis=0)
    idx = np.where(cur_poly[:, 0] * post_poly[:, 0] <= 0)
    idx = np.array(idx[0])
    p1 = cur_poly[idx]
    p2 = post_poly[idx]
    k = (p2[:, 1] - p1[:, 1]) / (p2[:, 0] - p1[:, 0] + 1e-8)
    b = p1[:, 1] - k * p1[:, 0]
    y_cx = np.round(b).astype(np.int32)
    y_cx = np.where(y_cx >= 0, -1e4, y_cx)
    if len(y_cx) > 0:
        c_idx = np.argmax(y_cx)
        f_id = idx[c_idx]
        n_poly = np.insert(poly, f_id + 1, [cx, cy + y_cx[c_idx]], axis=0)
        n_poly = np.roll(n_poly, -(f_id + 1), axis=0)
    else:
        n_poly = poly
    return n_poly

def compute_geosim(pred_poly, gt_poly):
    """Compute metrics, e.g. miou, weightcov, polysim

    Returns:
    polysim = sum(gt_size *(iou * geosim)) / gt_size
    geosim = 1 - (integral(min(|atan2(dg) - atan2(dp)|, |2PI + atan2(dg) - atan2(dp)|))) / 2PI
    """
    pred_poly = pred_poly.astype(np.int32)
    gt_poly = gt_poly.astype(np.int32)
    epsilon = 0.005 * cv2.arcLength(pred_poly[:, None], True)
    pred_approx = cv2.approxPolyDP(pred_poly[:, None], epsilon, True)
    if len(pred_approx) < 3 or len(gt_poly) < 3:
        return 0.0

    p_poly = rank_polygon(pred_approx[:, 0])
    t_poly = rank_polygon(gt_poly)

    # geosim
    cur_p_poly = p_poly
    post_p_poly = np.roll(cur_p_poly, -1, axis=0)
    p_v = post_p_poly - cur_p_poly
    p_d = np.sqrt(p_v[:, 0] ** 2 + p_v[:, 1] ** 2)
    p_s = np.cumsum(p_d) / np.sum(p_d)
    p_ang = np.arctan2(p_v[:, 0], p_v[:, 1])

    cur_t_poly = t_poly
    post_t_poly = np.roll(cur_t_poly, -1, axis=0)
    t_v = post_t_poly - cur_t_poly
    t_d = np.sqrt(t_v[:, 0] ** 2 + t_v[:, 1] ** 2)
    t_s = np.cumsum(t_d) / np.sum(t_d)
    t_ang = np.arctan2(t_v[:, 0], t_v[:, 1])

    union_s = np.union1d(p_s, t_s)
    u_p_ang = np.full_like(union_s, 1e4)
    for i in range(len(p_s)):
        idx = np.argwhere(union_s == p_s[i])[:, 0][0]
        u_p_ang[idx] = p_ang[i]
        u_p_ang[:idx] = np.where(u_p_ang[:idx] == 1e4, np.full_like(u_p_ang[:idx], p_ang[i]), u_p_ang[:idx])
    u_t_ang = np.full_like(union_s, 1e4)
    for i in range(len(t_s)):
        idx = np.argwhere(union_s == t_s[i])[:, 0][0]
        u_t_ang[idx] = t_ang[i]
        u_t_ang[:idx] = np.where(u_t_ang[:idx] == 1e4, np.full_like(u_t_ang[:idx], t_ang[i]), u_t_ang[:idx])
    cur_union_s = np.insert(union_s, 0, 0)
    post_union_s = np.roll(cur_union_s, -1, axis=0)
    union_d = (post_union_s - cur_union_s)[:-1]
    u_dist = np.minimum(np.abs(u_p_ang - u_t_ang), 2 * math.pi - np.abs(u_p_ang - u_t_ang))
    geo_sim = 1 - np.sum(union_d * u_dist) / (2 * math.pi)

    return geo_sim

def compute_metrics(gt_masks, pred_scores, pred_masks, gt_polys, pred_polys, iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    sum of recalls, precisions,... over per image
    """
    ap, ap50, ap75 = compute_ap_range(gt_masks, pred_scores, pred_masks)

    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(gt_masks, pred_scores, pred_masks, iou_threshold)
    gt_match = gt_match.astype(np.int32)

    c_gt_match, c_pred_match, c_overlaps = compute_matches2(gt_masks, pred_masks, 0.0)
    c_gt_match = c_gt_match.astype(np.int32)

    poly_sim_sum = 0
    gt_size_sum = 0

    for i in range(len(c_gt_match)):
        gt_poly = gt_polys[i].astype(np.int32)
        gt_size = np.sum((gt_masks[..., i] > .5).astype(np.float32))

        if c_gt_match[i] != -1:
            pred_poly = pred_polys[c_gt_match[i]].astype(np.int32)
            geo_sim = compute_geosim(pred_poly, gt_poly)
            iou = c_overlaps[c_gt_match[i], i]
            poly_sim_sum += geo_sim * iou * gt_size
        gt_size_sum += gt_size

    poly_sim_sum = round(poly_sim_sum, 6)
    gt_size_sum = round(gt_size_sum, 6)

    return ap, ap50, ap75, poly_sim_sum, gt_size_sum