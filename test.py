""" 
@author: Zhigang Jiang
@time: 2023/01/05
@description:
"""
import numpy as np


def calc_iou(gt_boxes: np.array, pred_boxes: np.array):
    """
    :param gt_boxes: [n_gt x 4(x1, y1, x2, y2)]
    :param pred_boxes: [m_pred x 4(x1, y1, x2, y2)]
    :return ious: [m_pred x n_gt]
    """

    n_gt = len(gt_boxes)
    m_pred = len(pred_boxes)

    ious = np.zeros((m_pred, n_gt))
    for i in range(m_pred):
        pred_box = pred_boxes[i]
        i_x_min = np.maximum(gt_boxes[:, 0], pred_box[0])
        i_y_min = np.maximum(gt_boxes[:, 1], pred_box[1])
        i_x_max = np.minimum(gt_boxes[:, 2], pred_box[2])
        i_y_max = np.minimum(gt_boxes[:, 3], pred_box[3])
        i_w = np.maximum(i_x_max - i_x_min, 0.)
        i_h = np.maximum(i_y_max - i_y_min, 0.)
        inters_area = i_w * i_h
        pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        ious[i] = inters_area / (pred_area + gt_areas - inters_area)

    return ious


def nms_cls(boxes, scores, threshold=0.2):
    """
    :param boxes: [n x 4(x1, y1, x2, y2)]
    :param scores: [n]
    :param threshold: iou threshold
    :return predicts_dict: processed by non-maximum suppression
    """
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        keep.append(order[0])
        iou = calc_iou(boxes[order[0]:order[0] + 1, :4], boxes[order[1:], :4])
        indexes = np.where(iou <= threshold)[0] + 1
        order = order[indexes]

    return boxes[keep]


def nms(predicts_dict, threshold=0.2):
    """
    :param predicts_dict: {"stick": [[x1, y1, x2, y2, scores1], [...]]}.
    :param threshold: iou threshold
    :return predicts_dict: processed by non-maximum suppression
    """
    for object_name, boxes in predicts_dict.items():
        predicts_dict[object_name] = nms_cls(boxes[:, :4],  boxes[:, 4], threshold)
    return predicts_dict


if __name__ == '__main__':
    nms({
        's': np.array([[1, 1, 3, 3, 0.8], [1, 1, 2, 2, 0.6], [2, 2, 8, 8, 0.9]])
    })