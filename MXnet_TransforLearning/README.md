## MXNet 实现简单的分类任务
首先做好图片文件夹，形式和tf版本一致：一个大文件夹下存放各个class的文件夹，每个class的文件夹内存放对应class的图片。<br>
然后，将`mxnet_classifiter.ipynb `数据预处理代码块中如下代码中的图片目录指定到自己的图片文件夹，<br>
```
train_ds = gdata.vision.ImageFolderDataset(root = r'.\Hotdog',  # 指定到图片根目录
                                           flag=1)              # 0转换为灰度图，1转换为彩色图
```
最后按顺序运行各个代码块即可。
