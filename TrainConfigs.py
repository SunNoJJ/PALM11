# -*- coding: UTF-8 -*-
"""
@Project ：PyramidBoxManyClass
@File ：Configs.py
@Author ：正途皆是道
@Date ：21-11-15 下午2:02
"""
import argparse
import functools
from utility import add_arguments

# 命令解析，源码中为了方便接收命令调用时设置的参数，此例直接修改即可。
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# 参数设置
# 训练数据父级位置
add_arg('data_dir', str, 'data/data117189/train_img', "The base dir of dataset")
add_arg('data_txt', str, 'data/data117189/train_img/PlamImg.txt', "The base dir of dataset")
# 模型有多少个目标分类
add_arg('class_num', int, 1, "模型分类数量")
# 模型保存位置
add_arg('model_save_dir', str, 'train_model/PLAM', "The path to save model.")
# 预训练模型位置
add_arg('pretrained_model', str, './train_model/PLAM/pre_model/', "The init model path.")
# 是否使用GPU
add_arg('use_gpu', bool, True, "Whether use GPU.")
# 每次训练几张图片
add_arg('batch_size', int, 4, "Minibatch size.")
# 所有图片训练几轮
add_arg('epoc_num', int, 600, "Epoch number.")
# 训练图片总数
add_arg('train_images', int, 800, "训练图片总数，结合lr,lr_epochs，lr_decay用来计算学习率的衰减步数 ")
# 学习率
add_arg('learning_rate', float, 0.0001, "The start learning rate.")
# 训练时学习率衰减歩长
# add_arg('lr_epochs', list, [10, 25, 40, 50, 55], "学习率衰减歩长")
add_arg('lr_epochs', list, [5, 10, 15, 20, 25], "学习率衰减歩长")
# 学习率衰减倍率,比歩长多1
add_arg('lr_decay', list, [1, 0.5, 0.1, 0.05, 0.01, 0.001], "学习率衰减倍率,比歩长多1")
# 是否使用GPU多线程
add_arg('parallel', bool, False, "Whether use multi-GPU/threads or not.")
# 是否使用pyramidbox
add_arg('use_pyramidbox', bool, True, "Whether use PyramidBox model.")
# 训练时图片的长宽，近似1080*1920的比例
add_arg('resize_h', int, 1280, "The resized image height.")
add_arg('resize_w', int, 1280, "The resized image width.")
# 通道转换比例
add_arg('mean_BGR', str, '18.,38.,69.', "Mean value for B,G,R channel which will be subtracted.")
# 是否使用多进程进行数据预处理
add_arg('use_multiprocess', bool, False, "Whether use multi-process for data preprocessing.")
parser.add_argument('--enable_ce', action='store_true', help='If set, run the task with continuous evaluation logs.')
parser.add_argument('--batch_num', type=int, help="batch num for ce")
parser.add_argument('--num_devices', type=int, default=1, help='Number of GPU devices')
