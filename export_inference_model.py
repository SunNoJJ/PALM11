# -*- coding: UTF-8 -*-
"""
@Project ：zzytgitee
@File ：export_inference_model.py
@Author ：正途皆是道
@Date ：21-8-19 下午3:26
"""
# 模型预测以及可视化

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import functools
import paddle

paddle.enable_static()
import paddle.fluid as fluid
from pyramidbox import PyramidBox
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
# 模型有多少个目标分类
add_arg('class_num', int, 7, "模型分类数量")
# 是否使用gpu
add_arg('use_gpu', bool, False, "Whether use GPU or not.")
# 是否使用金字塔箱模型
add_arg('use_pyramidbox', bool, True, "Whether use PyramidBox model.")
# 训练好的模型位置
add_arg('model_dir', str, './out/1', "The model path.")
# 输入图片的维度
add_arg('input_shape', list, [3, 640, 640], "input data shape")
# yapf: enable


if __name__ == '__main__':
    # 导入超参数
    args = parser.parse_args()
    print_arguments(args)

    class_num = args.class_num
    # 选择预测方式GPU or CPU
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    main_program = fluid.Program()
    startup_program = fluid.Program()
    image_shape = args.input_shape
    with fluid.program_guard(main_program, startup_program):
        # 加载网络
        network = PyramidBox(
            data_shape=image_shape,
            class_number=class_num,
            sub_network=args.use_pyramidbox,
            is_infer=True)
        infer_program, nmsed_out = network.infer(main_program)
        fetches = [nmsed_out]
        fluid.io.load_persistables(  # 加载持久性变量
            exe, args.model_dir, main_program=infer_program)
        # save model and program
        fluid.io.save_inference_model('infer_model', ['image'], [nmsed_out],
                                      exe, main_program=infer_program,
                                      params_filename='__params__')
        print('推理模型导出完成。。。。。。。。。。。')
