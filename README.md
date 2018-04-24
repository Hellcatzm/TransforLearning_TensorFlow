迁移学习TransforLearning
======================
[『TensorFlow』迁移学习](http://www.cnblogs.com/hellcat/p/6909269.html "我的博客")<br>
## 1、相关下载
数据和预训练模型下载：
```Shell
curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
wget https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip
```
## 2、项目简介
本项目将使用ImageNet数据集预训练好的InceptionV3网络结构舍弃后面全连接层，使用了新的分类器对花朵数据进行了迁移学习，迁移学习对于这种中等偏小的数据集又为合适。<br>

### 项目文件
`flower_photos`：文件目录，下面包含各个子类的`文件夹`，如果使用自己的数据的话，将自己数据各个类别分别放入一个文件夹，文件夹名字是类的字符串名字即可，将这些文件夹放入flower_photos文件夹内即可<br>
`TransforLearning.py`：主程序，用于训练，不过注意，可训练文件格式应该是jpg（jpeg、JPG等等写法均可）<br>
`TransferLearning_reload.py`：用于预测，仅能进行单张图片类别预测，需要进入文件中(21行左右)，将`image_path`修改为自己的图片路径<br>
其他文件夹为程序自己生成，不需要提前新建<br>
![](https://images2018.cnblogs.com/blog/1161096/201804/1161096-20180424094519006-1238870240.png "项目文件") 

### 运行命令
首先训练，
```Shell
python TransforLearning.py
```
等待训练完成后(不等也行，不过需要保证已经有训练中间生成模型被保存了)，预测一张自己的图片，
```Python
python TransferLearning_reload.py
```
命令很简单，之后就会输出预测信息，如下格式，
![](https://images2018.cnblogs.com/blog/1161096/201804/1161096-20180424094042927-662872256.png "分类信息") 
第一行表示分类的类别，这里是根据图片文件夹的名字来的，可以看到和之前的项目文件示意图中`flower_photos`的子文件夹名称一一对应；第二行为分类结果，每一个值和第一行对应位置的类别相对应，比如这个结果就是分类为daisy的概率为0.22。
