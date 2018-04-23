迁移学习TransforLearning
======================
[『TensorFlow』迁移学习_他山之石，可以攻玉](http://www.cnblogs.com/hellcat/p/6909269.html "我的博客")<br>
## 相关下载
数据和预训练模型下载：
```Shell
curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
wget https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip
```
## 项目简介
本项目将使用ImageNet数据集预训练好的InceptionV3网络结构舍弃后面全连接层，使用了新的分类器对花朵数据进行了迁移学习，迁移学习对于这种中等偏小的数据集又为合适。<br>

### 项目文件
flower_photos：文件目录，下面包含各个子类的`文件夹`，如果使用自己的数据的话，将自己数据各个类别分别放入一个文件夹，文件夹名字是类的字符串名字即可，将这些文件夹放入flower_photos文件夹内即可<br>
TransforLearning.py：主程序，用于训练，不过注意，可训练文件格式应该是jpg（jpeg、JPG等等写法均可）<br>
TransferLearning_reload.py：用于预测，仅能进行单张图片类别预测，需要进入文件中(21行左右)，将`image_path`修改为自己的图片路径<br>

### 运行命令
首先训练，
```Shell
python TransforLearning.py
```
等待训练完成后(不等也行，不过需要保证已经有训练中间生成模型被保存了)，预测一张自己的图片
```Python
python TransferLearning_reload.py
```
命令很简单，之后就会输出预测信息。
