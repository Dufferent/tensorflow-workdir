#!/usr/bin/python3.5
# conding=utf-8

# import pack
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
import numpy as np
import pathlib
import os
import random
import cv2 as cv

BATCH_SIZE=32
IMG_W=192
IMG_H=192

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # base下默认使用cpu
AUTOTUNE = tf.data.experimental.AUTOTUNE  # 最优化的线程数
# official function for images preprocession
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_W, IMG_H])
    image /= 255.0  # normalize to [0,1] range
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label

# my own test function
def Test_Img(ts_data_dir,prb_mdl,class_names):
    ts_root = pathlib.Path(ts_data_dir)
    ts_img_path = list(ts_root.glob("*"))
    ts_img_path = [str(path) for path in ts_img_path]
    ts_all = []
    for ts_img in ts_img_path:
        ts_raw = tf.io.read_file(ts_img)
        ts_tensor = tf.image.decode_image(ts_raw)
        ts_final = tf.image.resize(ts_tensor, [IMG_W, IMG_H])
        ts_final = ts_final / 255.0
        ts_final = (np.expand_dims(ts_final,0))
        ts_all.append(ts_final)
    # print (ts_all)
    for index in range(0,len(ts_all)):
        predict = prb_mdl.predict(ts_all[index])
        for i in range(0,len(class_names)):
            print (ts_img_path[index]," predict:")
            print (class_names[i],"probility:=>%lf"%predict[0][i])
        print (" ")


if __name__ == "__main__":
    data_route = input("请输入数据路径:\n")
    data_root  = pathlib.Path(data_route)
    # print (data_root)
    all_image_path = list(str(name) for name in data_root.glob("*/*"))
    random.shuffle(all_image_path)
    image_count = len(all_image_path)
    # print (image_count)
    # print (all_image_path)
    label_name = list(str(name).split('/')[1] for name in data_root.glob("*/"))
    label_name = sorted(label_name)
    # print (label_name)
    label_to_index = dict( (name,index) for index,name in enumerate(label_name))
    # car::0 people::1
    # print (label_to_index)
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] \
                                                for path in all_image_path]
    # print (all_image_labels[225:300])
    ds = tf.data.Dataset.from_tensor_slices((all_image_path, all_image_labels))
    image_label_ds = ds.map(load_and_preprocess_from_path_label)
    # print (image_label_ds)
    ds = image_label_ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(192, 192, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # 1:190/2=95x16->2:93/2x32->3:44/2x64 = 22 x 64->4:20/2x64
    model.add(layers.Flatten())
    model.add(layers.Dense(128,kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    # model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    # model.add(layers.Dense(30, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(layers.Dense(len(label_name)))

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    history = model.fit(ds, epochs=2,steps_per_epoch=int(image_count/BATCH_SIZE))
    prb_mdl = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
    Test_Img("./ts",prb_mdl,label_name)

