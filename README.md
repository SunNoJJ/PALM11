# AI-Studio-PALM眼底彩照中黄斑中央凹定位
> 预处理数据集：https://aistudio.baidu.com/aistudio/datasetdetail/117189  <br>
> 项目地址：https://aistudio.baidu.com/aistudio/projectdetail/3269726  <br>
# 训练  <br>
python train.py  <br>
# 预测  <br>
python widerface_eval_dir.py --image_dir=data/data117189/test_img/Testing400 --use_gpu=True --class_num=1 --model_dir=train_model/PLAM/pre_model --label_name_list='火''烟' --confs_threshold=0.23  <br>
## 项目描述
>  近视已成为全球公共卫生负担。在近视患者中，约35%为高度近视。近视导致眼轴长度的延长，可能引起视网膜和脉络膜的病理改变。随着近视屈光度的增加，高度近视将发展为病理性近视，其特点是病理改变的形成:(1)后极，包括镶嵌型眼底、后葡萄肿、视网膜脉络膜变性等;(2)视盘，包括乳头旁萎缩、倾斜等;(3)近视性黄斑，包括漆裂、福氏斑、CNV等。病理性近视对患者造成不可逆的视力损害。因此，早期诊断和定期随访非常重要。

## 项目结构
> 一目了然的项目结构能帮助更多人了解，目录树以及设计思想都很重要~
```
-|data
-|work
-README.MD
-xxx.ipynb
```
## 使用方式
> 相信你的Fans已经看到这里了，快告诉他们如何快速上手这个项目吧~  
A：在AI Studio上[运行本项目](https://aistudio.baidu.com/aistudio/projectdetail/3269726)  
B：此处由项目作者进行撰写使用方式。
