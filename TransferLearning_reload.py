# Author : hellcat
# Time   : 18-4-23

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
 
import numpy as np
np.set_printoptions(threshold=np.inf)
"""

import numpy as np
import pprint as pp
import tensorflow as tf
from TransforLearning import creat_image_lists

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

image_path = ['/home/hellcat/PycharmProjects/CV&DL/迁移学习/flower_photos/daisy/5673551_01d1ea993e_n.jpg',
              '/home/hellcat/PycharmProjects/CV&DL/迁移学习/flower_photos/roses/99383371_37a5ac12a3_n.jpg']

# with open(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:  # 阅读器上下文
#     graph_def = tf.GraphDef()  # 生成图
#     graph_def.ParseFromString(f.read())  # 图加载模型
# # 加载图上节点张量(按照句柄理解)
# bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(  # 从图上读取张量，同时导入默认图
#     graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])
#
# image_data = open(image_path, 'rb').read()
# bottleneck = sess.run(bottleneck_tensor, feed_dict={jpeg_data_tensor: image_data})

ckpt = tf.train.get_checkpoint_state('./model/')
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
saver.restore(sess, ckpt.model_checkpoint_path)
# pp.pprint(tf.get_default_graph().get_operations())
g = tf.get_default_graph()
for img in image_path:
    image_data = open(img, 'rb').read()
    bottleneck = sess.run(g.get_tensor_by_name('import/pool_3/_reshape:0'),
                          feed_dict={g.get_tensor_by_name('import/DecodeJpeg/contents:0'): image_data})

    class_result = sess.run(g.get_tensor_by_name('final_train_ops/Softmax:0'),
                            feed_dict={g.get_tensor_by_name('BottleneckInputPlaceholder:0'): bottleneck})

    images_lists = creat_image_lists(10, 10)
    tf.logging.info(images_lists.keys())
    print(np.squeeze(class_result))
