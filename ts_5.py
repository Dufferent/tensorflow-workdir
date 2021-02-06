import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv
import os
import pathlib
import random

AUTOTUNE = tf.data.experimental.AUTOTUNE

def VERSION():
    print (cv.__version__)
    print (tf.__version__)
    return

def preprocess_image(image):
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image, [160, 160])
    image /= 255.0

    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label

def change_range(image,label):
    return 2*image-1, label

def get_features(file_dir):
    data_root = pathlib.Path(file_dir)
    print (data_root)
    for item in data_root.iterdir():
        print (item)
    all_image_path = list(data_root.glob('*/*'))
    all_image_path = [str(path) for path in all_image_path]
    random.shuffle(all_image_path)
    image_count = len(all_image_path)
    print (image_count)
    print (all_image_path[:10])
    label_names = ["people","car"]
    label_to_index = dict((name,index) for index,name in enumerate(label_names))
    # print (label_to_index)
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_path]

    print("First 10 labels indices: ", all_image_labels[:50])
    for img_path in all_image_path:
        img_raw = tf.io.read_file(img_path)
        img_tensor = tf.image.decode_image(img_raw)
        #print (img_tensor.shape)
        #print (img_tensor.dtype)
        img_final = tf.image.resize(img_tensor, [160, 160])
        img_final = img_final / 255.0
        # print (img_final.numpy().min())
        # print (img_final.numpy().max())
        # print (img_final.shape)

    ds = tf.data.Dataset.from_tensor_slices((all_image_path, all_image_labels))

    image_label_ds = ds.map(load_and_preprocess_from_path_label)
    # print (image_label_ds)

    BATCH_SIZE = 32
    ds = image_label_ds.shuffle(buffer_size=image_count)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    # print (ds)

    mobile_net = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False)
    mobile_net.trainable=False
    keras_ds = ds.map(change_range)
    image_batch, label_batch = next(iter(keras_ds))
    feature_map_batch = mobile_net(image_batch)
    print (feature_map_batch.shape)

    model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(label_names), activation = 'softmax')])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])
    # model.summary()
    model.fit(ds, epochs=2, steps_per_epoch = 20)

    prb_mdl = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
    
    # ts_img_raw = tf.io.read_file("./ts/iron-man.jpg")
    # ts_img_tensor = tf.image.decode_image(ts_img_raw)
    # ts_img_final = tf.image.resize(ts_img_tensor, [160, 160])
    # ts_img_final = ts_img_final / 255.0

    # tmp_img = ts_img_final
    # print (tmp_img.shape)
    # tmp_img = (np.expand_dims(tmp_img,0))
    
    ts_root = pathlib.Path("/home/xny/tensorflow_workdir/ts")
    ts_img_path = list(ts_root.glob("*"))
    ts_img_path = [str(path) for path in ts_img_path]
    ts_all = []
    for ts_img in ts_img_path:
        ts_raw = tf.io.read_file(ts_img)
        ts_tensor = tf.image.decode_image(ts_raw)
        ts_final = tf.image.resize(ts_tensor, [160, 160])
        ts_final = ts_final / 255.0
        ts_final = (np.expand_dims(ts_final,0))
        ts_all.append(ts_final)
    print (ts_all)
    # predict = prb_mdl.predict(tmp_img)
    # tmps = 0
    # ct = 0
    # for i in range(0,2):
    #     if (tmps < predict[0][i]):
    #         tmps = predict[0][i]
    #         ct = i
    #     print (predict[0][i])
    # print ("single predict is:=>",class_names[ct])
    # print ("single real is:=>",class_names[0])
    ct = 0
    for ts in ts_all:
        predict = prb_mdl.predict(ts)
        for i in range(0,2):
            print (ts_img_path[ct]," predict:")
            print (class_names[i],"probility:=>%lf"%predict[0][i])
        print (" ")
        ct += 1


class_names = ["people","car"]

if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    sess = tf.compat.v1.Session(config=config)
    VERSION()
    get_features("/home/xny/tensorflow_workdir/data")