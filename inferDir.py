# -*- coding: UTF-8 -*-
"""
@Project ：zzytgitee
@File ：infer.py
@Author ：正途皆是道
@Date ：21-8-19 下午3:49
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import PIL
import numpy as np
import argparse
import functools
from PIL import Image
import paddle.fluid as fluid
import reader
from visualize import draw_bboxes
from utility import add_arguments, print_arguments
import paddle

paddle.enable_static()
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
# 需要先运行export_inference_model.py导出推理模型
# 模型有多少个目标分类
add_arg('class_num', int, 7, "模型分类数量")
# 是否使用gpu
add_arg('use_gpu', bool, False, "Whether use GPU or not.")
# 是否使用金字塔箱模型
add_arg('use_pyramidbox', bool, True, "Whether use PyramidBox model.")
# 阈值
add_arg('confs_threshold', float, 0.65, "Confidence threshold to draw bbox.")
# 预测图片路径
add_arg('image_dir', str, './test_img', "The image used to inference and visualize.")
# 导出推理模型的位置
add_arg('infer_model', str, './infer_model', "The infer model path.")
# 训练图片的BGR平均值
add_arg('mean_bgr', list, [104., 117., 123.], "The img B,G,R mean")
# yapf: enable

def infer(args):
    image_dir = args.image_dir
    img_ns = os.listdir(image_dir)
    for img_n in img_ns:
        image_path = os.path.join(args.image_dir, img_n)
        image = Image.open(image_path)
        if type(image) != PIL.JpegImagePlugin.JpegImageFile and type(image) != PIL.PngImagePlugin.PngImageFile:
            print('请输入png或者jpg文件')
            continue
        if image.mode == 'L':
            image = image.convert('RGB')
        shrink, max_shrink = get_shrink(image.size[1], image.size[0])

        det0 = detect_face(image, shrink)
        if args.use_gpu:
            det1 = flip_test(image, shrink)
            [det2, det3] = multi_scale_test(image, max_shrink)
            det4 = multi_scale_test_pyramid(image, max_shrink)
            det = np.row_stack((det0, det1, det2, det3, det4))
            dets = bbox_vote(det)
        else:
            # when infer on cpu, use a simple case
            dets = det0
        # print(dets)
        keep_index = np.where(dets[:, 4] >= args.confs_threshold)[0]
        boxs = dets[keep_index, :]
        score = boxs[:, 4]
        labels = boxs[:, 5]
        # label_name = ['壹','贰','叁','肆','伍','陆','柒','捌','玖','拾']
        label_name = ['红色', '绿色', '蓝色', '白色', '黄色', '紫色', '青色', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖', '拾']
        draw_bboxes(image_path, boxs[:, 0:4], labels, score, label_name=label_name)


def save_widerface_bboxes(image_path, bboxes_scores, output_dir):
    """
    Save predicted results, including bbox and score into text file.
    Args:
        image_path (string): file name.
        bboxes_scores (np.array|list): the predicted bboxed and scores, layout
            is (xmin, ymin, xmax, ymax, score)
        output_dir (string): output directory.
    """
    image_name = image_path.split('/')[-1]
    image_class = image_path.split('/')[-2]

    odir = os.path.join(output_dir, image_class)
    if not os.path.exists(odir):
        os.makedirs(odir)

    ofname = os.path.join(odir, '%s.txt' % (image_name[:-4]))
    f = open(ofname, 'w')
    f.write('{:s}\n'.format(image_class + '/' + image_name))
    f.write('{:d}\n'.format(bboxes_scores.shape[0]))
    for box_score in bboxes_scores:
        xmin, ymin, xmax, ymax, score = box_score
        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(xmin, ymin, (
                xmax - xmin + 1), (ymax - ymin + 1), score))
    f.close()
    print("The predicted result is saved as {}".format(ofname))


def detect_face(image, shrink):
    image_shape = [3, image.size[1], image.size[0]]
    if shrink != 1:
        h, w = int(image_shape[1] * shrink), int(image_shape[2] * shrink)
        image = image.resize((w, h), Image.ANTIALIAS)
        image_shape = [3, h, w]

    img = np.array(image)
    img = reader.to_chw_bgr(img)
    mean = [np.mean(img[0]), np.mean(img[1]), np.mean(img[2])]
    scale = 0.007843
    img = img.astype('float32')
    img -= np.array(mean)[:, np.newaxis, np.newaxis].astype('float32')
    img = img * scale
    img = [img]
    img = np.array(img)

    detection, = exe.run(infer_program,
                         feed={'image': img},
                         fetch_list=fetches,
                         return_numpy=False)
    detection = np.array(detection)
    # layout: xmin, ymin, xmax. ymax, score
    if np.prod(detection.shape) == 1:
        print("No face detected")
        return np.array([[0, 0, 0, 0, 0, 0]])

    det_label = detection[:, 0]
    det_conf = detection[:, 1]
    det_xmin = image_shape[2] * detection[:, 2] / shrink
    det_ymin = image_shape[1] * detection[:, 3] / shrink
    det_xmax = image_shape[2] * detection[:, 4] / shrink
    det_ymax = image_shape[1] * detection[:, 5] / shrink

    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf, det_label))
    return det


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    if det.shape[0] == 0:
        dets = np.array([[10, 10, 20, 20, 0.002]])
        det = np.empty(shape=[0, 5])
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # nms
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)
        if merge_index.shape[0] <= 1:
            if det.shape[0] == 0:
                try:
                    dets = np.row_stack((dets, det_accu))
                except:
                    dets = det_accu
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        label = np.max(det_accu[:, 5])
        det_accu_sum = np.zeros((1, 6))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4],
                                      axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        det_accu_sum[:, 5] = label
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum
    dets = dets[0:750, :]
    return dets


def flip_test(image, shrink):
    img = image.transpose(Image.FLIP_LEFT_RIGHT)
    det_f = detect_face(img, shrink)
    det_t = np.zeros(det_f.shape)
    # image.size: [width, height]
    det_t[:, 0] = image.size[0] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.size[0] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def multi_scale_test(image, max_shrink):
    # Shrink detecting is only used to detect big faces
    st = 0.5 if max_shrink >= 0.75 else 0.5 * max_shrink
    det_s = detect_face(image, st)
    index = np.where(
        np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1)
        > 30)[0]
    det_s = det_s[index, :]
    # Enlarge one times
    bt = min(2, max_shrink) if max_shrink > 1 else (st + max_shrink) / 2
    det_b = detect_face(image, bt)

    # Enlarge small image x times for small faces
    if max_shrink > 2:
        bt *= 2
        while bt < max_shrink:
            det_b = np.row_stack((det_b, detect_face(image, bt)))
            bt *= 2
        det_b = np.row_stack((det_b, detect_face(image, max_shrink)))

    # Enlarged images are only used to detect small faces.
    if bt > 1:
        index = np.where(
            np.minimum(det_b[:, 2] - det_b[:, 0] + 1,
                       det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    # Shrinked images are only used to detect big faces.
    else:
        index = np.where(
            np.maximum(det_b[:, 2] - det_b[:, 0] + 1,
                       det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]
    return det_s, det_b


def multi_scale_test_pyramid(image, max_shrink):
    # Use image pyramids to detect faces
    det_b = detect_face(image, 0.25)
    index = np.where(
        np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1)
        > 30)[0]
    det_b = det_b[index, :]

    st = [0.75, 1.25, 1.5, 1.75]
    for i in range(len(st)):
        if (st[i] <= max_shrink):
            det_temp = detect_face(image, st[i])
            # Enlarged images are only used to detect small faces.
            if st[i] > 1:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) < 100)[0]
                det_temp = det_temp[index, :]
            # Shrinked images are only used to detect big faces.
            else:
                index = np.where(
                    np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) > 30)[0]
                det_temp = det_temp[index, :]
            det_b = np.row_stack((det_b, det_temp))
    return det_b


def get_shrink(height, width):
    """
    Args:
        height (int): image height.
        width (int): image width.
    """
    # avoid out of memory
    max_shrink_v1 = (0x7fffffff / 577.0 / (height * width)) ** 0.5
    max_shrink_v2 = ((678 * 1024 * 2.0 * 2.0) / (height * width)) ** 0.5

    def get_round(x, loc):
        str_x = str(x)
        if '.' in str_x:
            str_before, str_after = str_x.split('.')
            len_after = len(str_after)
            if len_after >= 3:
                str_final = str_before + '.' + str_after[0:loc]
                return float(str_final)
            else:
                return x

    max_shrink = get_round(min(max_shrink_v1, max_shrink_v2), 2) - 0.3
    if max_shrink < 0:
        max_shrink = max_shrink + 0.3
    if max_shrink >= 1.5 and max_shrink < 2:
        max_shrink = max_shrink - 0.1
    elif max_shrink >= 2 and max_shrink < 3:
        max_shrink = max_shrink - 0.2
    elif max_shrink >= 3 and max_shrink < 4:
        max_shrink = max_shrink - 0.3
    elif max_shrink >= 4 and max_shrink < 5:
        max_shrink = max_shrink - 0.4
    elif max_shrink >= 5:
        max_shrink = max_shrink - 0.5

    shrink = max_shrink if max_shrink < 1 else 1
    return shrink, max_shrink


if __name__ == '__main__':
    # 导入超参数
    args = parser.parse_args()
    print_arguments(args)
    class_num = args.class_num
    # 选择预测方式GPU or CPU
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # 加载模型参数
    path = args.infer_model
    [infer_program, feed_target_names, fetches] = fluid.io.load_inference_model(dirname=path, executor=exe,
                                                                                params_filename='__params__')

    # 预测
    infer(args)
