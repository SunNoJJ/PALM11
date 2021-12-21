# -*- coding: UTF-8 -*-
"""
@Project ：zzytgitee
@File ：test.py
@Author ：正途皆是道
@Date ：21-6-22 下午5:23
"""
'''
## self._vgg()
    self.conv3, self.pool3
    self.conv4, self.pool4
    self.conv5, self.pool5
    self.conv6, self.pool6
    self.conv7, self.pool7
    self.conv8, self.pool8
## self._low_level_fpn()
self.lfpn2_on_conv5 = fpn(self.conv6, self.conv5)
self.lfpn1_on_conv4 = fpn(self.lfpn2_on_conv5, self.conv4)
self.lfpn0_on_conv3 = fpn(self.lfpn1_on_conv4, self.conv3)    
## self._cpm_module()
self.ssh_conv3 = cpm(self.lfpn0_on_conv3)
self.ssh_conv4 = cpm(self.lfpn1_on_conv4)
self.ssh_conv5 = cpm(self.lfpn2_on_conv5)
self.ssh_conv6 = cpm(self.conv6)
self.ssh_conv7 = cpm(self.conv7)
self.ssh_conv8 = cpm(self.conv8)
## self._pyramidbox()
self.ssh_conv3_norm = self._l2_norm_scale(self.ssh_conv3, init_scale=10.)
self.ssh_conv4_norm = self._l2_norm_scale(self.ssh_conv4, init_scale=8.)
self.ssh_conv5_norm = self._l2_norm_scale(self.ssh_conv5, init_scale=5.)

self.face_mbox_loc = fluid.layers.concat(face_locs, axis=1)
self.face_mbox_conf = fluid.layers.concat(face_confs, axis=1)

self.head_mbox_loc = fluid.layers.concat(head_locs, axis=1)
self.head_mbox_conf = fluid.layers.concat(head_confs, axis=1)

self.prior_boxes = fluid.layers.concat(boxes)
self.box_vars = fluid.layers.concat(vars)

## train()
face_loss = fluid.layers.ssd_loss(
            self.face_mbox_loc,#位置预测[N,M,4],M是输入的预测bounding box的个数，N是batch size，
            self.face_mbox_conf,#置信度(分类)预测[N,M,C],C=类别数量+1,有个0类,每个类别有上下文类
            self.face_box,#真实框(bbox),
            self.gt_label,#ground-truth标签
            self.prior_boxes,# 检测网络生成的候选框,
            self.box_vars,# 候选框的方差
            )
head_loss = fluid.layers.ssd_loss(
            self.head_mbox_loc,
            self.head_mbox_conf,
            self.head_box,
            self.gt_label,
            self.prior_boxes,
            self.box_vars)   
#reduce_sum对Tensor进行求和运算
face_loss = fluid.layers.reduce_sum(face_loss)
head_loss = fluid.layers.reduce_sum(head_loss)
total_loss = face_loss + head_loss
return face_loss, head_loss, total_loss  
## infer()
1.根据先验框(prior_box)信息和回归位置偏移解码出预测框坐标。
2.通过多类非极大值抑制(NMS)获得最终检测输出。
face_nmsed_out = fluid.layers.detection_output(
                self.face_mbox_loc, #回归位置偏移,[N,M,4]，M是输入的预测bounding box的个数，N是batch size，每个bounding box有四个坐标值
                self.face_mbox_conf,#未归一化的置信度。维度为[N,M,C]，N和M的含义同上，C是类别数
                self.prior_boxes,   #先验框,维度为[M,4]，M是提取的先验框个数，格式为[xmin,ymin,xmax,ymax]
                self.box_vars,      #先验框的方差，和 prior_box 维度相同
                nms_threshold=0.3,
                nms_top_k=5000,
                keep_top_k=750,
                score_threshold=0.01)   
                     

face_nmsed_out # [label, confidence, xmin, ymin, xmax, ymax]
'''



