# -*- coding: UTF-8 -*-
"""
@Project ：zzytgitee
@File ：train.py
@Author ：正途皆是道
@Date ：21-5-19 下午3:40
"""
# 训练代码
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import numpy as np
import time

import paddle

paddle.enable_static()


def set_paddle_flags(**kwargs):
    for key, value in kwargs.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


# 以上设置需要在导入paddle之前设置，否则无效
set_paddle_flags(
    FLAGS_eager_delete_tensor_gb=0,  # 启用GC保存内存
)

import paddle.fluid as fluid
from pyramidbox import PyramidBox
import reader
from utility import print_arguments, check_cuda
from TrainConfigs import parser

# 训练参数设置
train_parameters = {
    "train_images": 9999999999,
    "image_shape": [],  # [3, 640, 640],
    "class_num": 9999999999,
    "batch_size": 9999999999,
    "lr": 9999999999,  # 0.001,
    "lr_epochs": [],  # [99, 124, 149],
    "lr_decay": [],  # [1, 0.1, 0.01, 0.001],
    "epoc_num": 0,  # 160,
    "optimizer_method": "momentum",
    "use_pyramidbox": True
}


# 优化器设置
def optimizer_setting(train_params):
    batch_size = train_params["batch_size"]
    iters = train_params["train_images"] // batch_size
    lr = train_params["lr"]
    optimizer_method = train_params["optimizer_method"]
    boundaries = [i * iters for i in train_params["lr_epochs"]]
    values = [i * lr for i in train_params["lr_decay"]]
    """学习率衰减方式
    boundaries = [10000, 20000]
    values = [1.0, 0.5, 0.1]
    if step < 10000:
        learning_rate = 1.0
    elif 10000 <= step < 20000:
        learning_rate = 0.5
    else:
        learning_rate = 0.1
    """
    if optimizer_method == "momentum":
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(boundaries, values),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(0.0005),
        )
    else:
        optimizer = fluid.optimizer.RMSProp(
            learning_rate=fluid.layers.piecewise_decay(boundaries, values),
            regularization=fluid.regularizer.L2Decay(0.0005),
        )
    return optimizer


def build_program(train_params, main_prog, startup_prog, args):
    use_pyramidbox = train_params["use_pyramidbox"]
    image_shape = train_params["image_shape"]
    class_num = train_params["class_num"]
    with fluid.program_guard(main_prog, startup_prog):
        py_reader = fluid.layers.py_reader(
            capacity=8,
            shapes=[[-1] + image_shape, [-1, 4], [-1, 4], [-1, 1]],
            lod_levels=[0, 1, 1, 1],
            dtypes=["float32", "float32", "float32", "int32"],
            use_double_buffer=True)
        with fluid.unique_name.guard():
            image, face_box, head_box, gt_label = fluid.layers.read_file(py_reader)
            fetches = []
            network = PyramidBox(image=image,
                                 face_box=face_box,
                                 head_box=head_box,
                                 gt_label=gt_label,
                                 class_number=class_num,
                                 sub_network=use_pyramidbox)
            if use_pyramidbox:
                face_loss, head_loss, loss = network.train()
                fetches = [face_loss, head_loss]
            else:
                loss = network.vgg_ssd_loss()
                fetches = [loss]
            optimizer = optimizer_setting(train_params)
            optimizer.minimize(loss)
    return py_reader, fetches, loss


# 训练函数
def train(args, config, train_params, train_file_list):
    batch_size = train_params["batch_size"]  # 每块大小
    epoc_num = train_params["epoc_num"]  # 训练次数
    optimizer_method = train_params["optimizer_method"]  # 优化方法
    use_pyramidbox = train_params["use_pyramidbox"]  # 是否使用金字塔箱模型

    use_gpu = args.use_gpu  # 是否使用gpu
    model_save_dir = os.path.join(args.model_save_dir, '')
    pretrained_model = args.pretrained_model

    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""  # 获取GPU设备信息
    devices_num = len(devices.split(","))  # GPU个数
    # 一些简单计算
    batch_size_per_device = batch_size // devices_num
    iters_per_epoc = train_params["train_images"] // batch_size
    num_workers = 8
    is_shuffle = True

    startup_prog = fluid.Program()
    train_prog = fluid.Program()

    # only for ce
    if args.enable_ce:
        SEED = 102
        startup_prog.random_seed = SEED
        train_prog.random_seed = SEED
        num_workers = 1
        pretrained_model = ""
        if args.batch_num != None:
            iters_per_epoc = args.batch_num

    train_py_reader, fetches, loss = build_program(
        train_params=train_params,
        main_prog=train_prog,
        startup_prog=startup_prog,
        args=args)

    # 使用GPU or CPU
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    start_epoc = 0
    # 使用预训练模型
    if pretrained_model:
        if pretrained_model.isdigit():
            start_epoc = int(pretrained_model) + 1
            pretrained_model = os.path.join(model_save_dir, pretrained_model)
            print("Resume from %s " % (pretrained_model))

        if not os.path.exists(pretrained_model):
            raise ValueError("The pre-trained model path [%s] does not exist." %
                             (pretrained_model))

        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(
            exe, pretrained_model, main_program=train_prog, predicate=if_exist)

    # 开始训练
    train_reader = reader.train(config,
                                train_file_list,
                                batch_size_per_device,
                                shuffle=is_shuffle,
                                use_multiprocess=args.use_multiprocess,
                                num_workers=num_workers)
    # reader.decorate_paddle_reader(
    #     paddle.reader.shuffle(
    #     paddle.batch(mnist.train(), batch_size=5),
    #                               buf_size=1000)
    #                           )
    train_py_reader.decorate_paddle_reader(train_reader)

    # 多GPU
    if args.parallel:
        train_exe = fluid.ParallelExecutor(main_program=train_prog, use_cuda=use_gpu, loss_name=loss.name)

    # 模型保存
    def save_model(postfix, program):
        model_path = os.path.join(model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)

        print('save models to %s' % (model_path))
        fluid.io.save_persistables(exe, model_path, main_program=program)

    # 计算每次，每块训练结果 ，每训练十块打印一次训练结果
    # 结果有训练次数
    # 训练块数
    # 平均头部、脸部损失
    # 训练每块时间
    # 总时间
    total_time = 0.0
    epoch_idx = 0
    face_loss = 0
    head_loss = 0
    for pass_id in range(start_epoc, epoc_num):
        epoch_idx += 1
        start_time = time.time()
        epoch_sart_time = time.time()
        prev_start_time = start_time
        end_time = 0
        batch_id = 0
        train_py_reader.start()
        while True:
            try:
                prev_start_time = start_time
                start_time = time.time()
                if args.parallel:
                    fetch_vars = train_exe.run(fetch_list=[v.name for v in fetches])
                else:
                    fetch_vars = exe.run(train_prog, fetch_list=fetches)
                end_time = time.time()
                fetch_vars = [np.mean(np.array(v)) for v in fetch_vars]
                face_loss = fetch_vars[0]
                head_loss = fetch_vars[1]
                if batch_id % 10 == 0:
                    with open('PramidLoss.txt', 'a+', encoding='utf-8') as save_loss:
                        save_loss.write(str(face_loss) + '\r\n')
                    if not args.use_pyramidbox:
                        print("Epoc {:d}, Batch {:d}, Loss {:.6f}, Time {:.5f}".format(
                            pass_id, batch_id, face_loss,
                            start_time - prev_start_time))
                    else:
                        print("Epoc {:d}, Batch {:d}, Pramid loss {:.6f}, " \
                              "Subsidiary loss {:.6f}, " \
                              "Time {:.5f}".format(pass_id,
                                                   batch_id, face_loss, head_loss,
                                                   start_time - prev_start_time))
                batch_id += 1
            except (fluid.core.EOFException, StopIteration):
                train_py_reader.reset()
                break
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_sart_time
        total_time += epoch_end_time - start_time
        print('第{:0>3}回训练用时{:.5f}秒'.format(pass_id,epoch_time))
        save_model(str(pass_id), train_prog)
        os.system("mv {0}pre_model {0}pre_model_bak".format(model_save_dir))  # 先备份
        os.system("mv {0}{1} {0}pre_model".format(model_save_dir, pass_id))  # 然后重命名
        os.system("rm -rf {}pre_model_bak".format(model_save_dir))  # 再删除

    # only for ce
    if args.enable_ce:
        gpu_num = get_cards(args)
        print("kpis\teach_pass_duration_card%s\t%s" %
              (gpu_num, total_time / epoch_idx))
        print("kpis\ttrain_face_loss_card%s\t%s" %
              (gpu_num, face_loss))
        print("kpis\ttrain_head_loss_card%s\t%s" %
              (gpu_num, head_loss))


def get_cards(args):
    if args.enable_ce:
        cards = os.environ.get('CUDA_VISIBLE_DEVICES')
        num = len(cards.split(","))
        return num
    else:
        return args.num_devices


if __name__ == '__main__':
    # 导入超参数
    args = parser.parse_args(args=[])  # 源码中没有参数，但没有参数会报错，具体原因待学习
    print_arguments(args)
    check_cuda(args.use_gpu)

    mean_BGR = [float(m) for m in args.mean_BGR.split(",")]  # 获取各通道占比
    image_shape = [3, int(args.resize_h), int(args.resize_w)]  # 图片形状设置
    train_parameters["image_shape"] = image_shape
    train_parameters["use_pyramidbox"] = args.use_pyramidbox
    train_parameters["batch_size"] = args.batch_size
    train_parameters["lr"] = args.learning_rate
    train_parameters["epoc_num"] = args.epoc_num
    train_parameters["class_num"] = args.class_num
    train_parameters["train_images"] = args.train_images
    train_parameters["lr_epochs"] = args.lr_epochs
    train_parameters["lr_decay"] = args.lr_decay

    # 数据加载
    # data_dir = os.path.join(args.data_dir, 'color_img/')
    # train_file_list = os.path.join(args.data_dir, 'color_mask_7.txt')
    data_dir = args.data_dir
    train_file_list = args.data_txt

    # 训练配置
    config = reader.Settings(
        data_dir=data_dir,
        resize_h=image_shape[1],
        resize_w=image_shape[2],
        apply_distort=True,
        apply_expand=False,
        mean_value=mean_BGR,
        ap_version='11point')
    print('开始训练')
    train(args, config, train_parameters, train_file_list)
    print('finish')
