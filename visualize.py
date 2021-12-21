#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import cv2 as cv
import numpy as np


def draw_bboxes(image_file, bboxes, labels=None, score=None, output_dir='./infer_result/', label_name=[]):
    """
    Draw bounding boxes on image.
    Args:
        image_file (string): input image path.
        bboxes (np.array): bounding boxes.
        labels (list of string): the label names of bboxes.
        output_dir (string): output directory.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    image = Image.open(image_file)
    img_h,img_w = image.size[1],image.size[0]
    draw = ImageDraw.Draw(image)
    for i in range(len(bboxes)):
        xmin, ymin, xmax, ymax = bboxes[i]
        if xmin<0 or ymin<0 or xmax<0 or ymax<0 or xmin>img_w or ymin>img_h or xmax>img_w or  ymax>img_h:
            continue
        area = abs((xmax-xmin)* (ymin-ymax))
        if area>100000:  # 保留小于阈值的面积
            continue
        print(area,end=' ')
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
        draw.line(
            [(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)], width=4, fill='green')
        # if labels and image.mode == 'RGB':
        kumo_font = ImageFont.truetype(r'./simfang.ttf', size=40)
        draw.text((left + 10, top + 2), label_name[int(labels[i]) - 1] + '\n__{:.4f}'.format(score[i]), (255, 0, 255),
                  font=kumo_font)
    output_file = image_file.split('/')[-1].replace('.jpg', '.png')
    if output_dir:
        output_file = os.path.join(output_dir, output_file)
    print("The image with bbox is saved as {}".format(output_file))
    image.save(output_file)

def writer_csv_box(image_file, bboxes, label_name=[]):
    image = Image.open(image_file)
    img_h,img_w = image.size[1],image.size[0]
    imgName, Fovea_X, Fovea_Y = image_file.split('/')[-1], 0, 0
    if len(bboxes) > 0:
        # 保留大于0的坐标
        # xmin, ymin, xmax, ymax, score, label
        # 0<xmin<img_w
        xmin_gt_0 = bboxes[np.argwhere(bboxes[:, 0] > 0).reshape((-1)), :]
        xmin_gt_0 = xmin_gt_0[np.argwhere(xmin_gt_0[:, 0]<img_w).reshape((-1)), :]
        # 0<xmax<img_w
        xmax_gt_0 = xmin_gt_0[np.argwhere(xmin_gt_0[:, 2] > 0 ).reshape((-1)), :]
        xmax_gt_0 = xmax_gt_0[np.argwhere(xmax_gt_0[:, 2]<img_w).reshape((-1)), :]
        # 0<ymin<img_h
        ymin_gt_0 = xmax_gt_0[np.argwhere(xmax_gt_0[:, 1] > 0).reshape((-1)), :]
        ymin_gt_0 = ymin_gt_0[np.argwhere(ymin_gt_0[:, 1]<img_h).reshape((-1)), :]
        # 0<ymax<img_h
        ymax_gt_0 = ymin_gt_0[np.argwhere(ymin_gt_0[:, 1] > 0).reshape((-1)), :]
        ymax_gt_0 = ymax_gt_0[np.argwhere(ymax_gt_0[:, 1]<img_h).reshape((-1)), :]
        # 保留小于阈值的面积
        area = 10000
        area_gt_200 = ymax_gt_0[np.argwhere(abs((ymax_gt_0[:,0]-ymax_gt_0[:,2])*(ymax_gt_0[:,1]-ymax_gt_0[:,3])) < area ).reshape((-1)),:]
        result = list(area_gt_200)
        result.sort(key=lambda x:x[4],reverse=True)
        if len(result) > 0:
            save_np = result[0]
            Fovea_X, Fovea_Y = [abs(save_np[2] + save_np[0])/2, abs(save_np[3] + save_np[1])/2]
    with open('./Fovea_Localization_Results.csv', 'a+', encoding='utf-8-sig', newline='') as csf:
        print(image_file)
        writer = csv.writer(csf)
        # write_data = ['imgName', 'Fovea_X', 'Fovea_Y']
        write_data = [imgName, str(Fovea_X), str(Fovea_Y)]
        writer.writerow(write_data)

def save_bbox_as_img(image_file, bboxes, labels=None, label_name=[]):
    """
    保存目标框为图片
    :param image_file:
    :param bboxes:
    :param labels:
    :param label_name:
    :return:
    """
    img_np = cv.imread(image_file)  # hwc
    for i in range(len(bboxes)):
        xmin, ymin, xmax, ymax = bboxes[i]
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        box_label = str(int(labels[i]))
        box_np = img_np[ymin:ymax, xmin:xmax, :]
        save_name = image_file.split('/')[-1].replace('.jpg', '') + '_xywh_'
        box_list = [xmin, ymin, xmax - xmin, ymax - ymin]
        save_name += "_".join('{}'.format(id) for id in box_list)
        save_name += '_' + box_label + '_.png'
        save_path = '/media/yt/data/LTL20210529/dataset/车牌/box_save/' + save_name
        cv.imwrite(save_path, box_np)

