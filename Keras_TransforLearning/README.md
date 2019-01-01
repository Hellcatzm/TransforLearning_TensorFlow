## Keras 实现简单的分类任务
实际上对于基础的分类任务，不论是迁移学习特征提取器或者时对网络整体进行微调，Keras都可以很轻易地实现。得益于Keras Module的方便接口，迁移学习可以很轻松的设定特征网络参数是否冻结，适应不同的数据集。<br>
首先做好图片文件夹，形式和tf版本一致：一个大文件夹下存放各个class的文件夹，每个class的文件夹内存放对应class的图片。
然后，将`keras_classifiter.ipynb`数据生成器块中如下代码中的图片目录指定到自己的图片文件夹，
```
train_flow = train_datagen.flow_from_directory(
        r'.\猫狗数据',            # this is the target directory
        target_size=(224, 224),  # all images will be resized to 150x150
        batch_size=32,
        class_mode='categorical')
```
注意，数据生成器块我定义了两种生成方式：自建的生成器或者官方API，选择其一即可，推荐官方，自建的我只考虑了二分类的情况，对于多分类需要自行修改代码。
