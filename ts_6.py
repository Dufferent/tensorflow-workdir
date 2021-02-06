import tensorflow as tf
from tensorflow import keras
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import time
import random

#offical define 
AUTOTUNE = tf.data.experimental.AUTOTUNE

#offical define 

#User define 
IMG_WIDTH = 192
IMG_HEGHT = 192
IMG_BATCH = 32
#User define 

#GPU SETs
def GPU_SET():
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    sess = tf.compat.v1.Session(config=config)
#GPU SETs

def VERSION():
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

def Show_Img(img_dir,index):
    data_dir = pathlib.Path(img_dir)
    data_abs_dir = list(data_dir.glob("*"))
    data_abs_dir = [str(path) for path in data_abs_dir]
    print (data_abs_dir)
    img = cv.imread(data_abs_dir[index],0)
    cv.imshow("img",img)
    cv.waitKey(0)

def Process_Img(path,img_height,img_width):
    for img in path:
        img_raw = tf.io.read_file(img)
        img_tensor = tf.image.decode_image(img_raw)
        img_final = tf.image.resize(img_tensor,[img_height,img_width])
        img_final = img_final / 255.0

# offical functions
def preprocess_image(image, w, h):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [w, h])
    image /= 255.0 
    # make the range to [0,1]
    return image

def load_and_preprocess_image(path, w, h):
    image = tf.io.read_file(path)
    return preprocess_image(image , w, h)

def load_and_preprocess_from_path_label(path, label, w=IMG_WIDTH, h=IMG_HEGHT):
    return load_and_preprocess_image(path,w,h), label

def change_range(image,label):
    return 2*image-1, label

# offical functions

def Create_Ds(data_dir,batch_size):
    data_root = pathlib.Path(data_dir)
    img_path = list(data_root.glob("*/*"))
    img_path = [str(path) for path in img_path]
    # for var in img_path:
    #     print (var)
    # print (len(img_path))
    img_count = len(img_path)
    random.shuffle(img_path)
    # for var in img_path:
    #     print (var)
    label_tmps = list(data_root.glob("*/"))
    label_tmps = [str(path) for path in label_tmps]
    labels_names = []
    for var in label_tmps:
        tmps = var.split('/')
        labels_names.append(tmps[len(tmps)-1])
    # print (labels_names)
    label_2_index = dict( (name,index) for index,name in enumerate(labels_names))
    # print (label_2_index)
    img_labels = [label_2_index[pathlib.Path(path).parent.name] for path in img_path]
    # print (img_labels[:10])
    # Process_Img(img_path,IMG_HEIGHT,IMG_WIDTH)
    ds = tf.data.Dataset.from_tensor_slices((img_path,img_labels))
    ds_combine = ds.map(load_and_preprocess_from_path_label)
    # print (ds_combine)
    # ds = ds_combine.shuffle(buffer_size = img_count)
    # ds = ds.repeat()
    # ds = ds.batch(batch_size)
    # ds = ds.prefetch(buffer_size = AUTOTUNE)
    
    ds = ds_combine.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=img_count))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    # print (ds)
    keras_ds = ds.map(change_range)
    return keras_ds,labels_names,img_count

def Create_Neural_Network(ds,image_batch,labels_names):
    mobile_net = tf.keras.applications.MobileNetV2(input_shape=(IMG_WIDTH, IMG_HEGHT, 3), include_top=False)
    mobile_net.trainable=False
    # make the expect 2 [-1,1]
    img_batch,label_batch = next(iter(ds))
    feature_map_batch = mobile_net(img_batch)
    # print(feature_map_batch.shape)
    model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(labels_names), activation = 'softmax')])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss='sparse_categorical_crossentropy',
                metrics=["accuracy"])
    # model.summary()
    return model

def Test_Img(ts_data_dir,prb_mdl,class_names):
    ts_root = pathlib.Path(ts_data_dir)
    ts_img_path = list(ts_root.glob("*"))
    ts_img_path = [str(path) for path in ts_img_path]
    ts_all = []
    for ts_img in ts_img_path:
        ts_raw = tf.io.read_file(ts_img)
        ts_tensor = tf.image.decode_image(ts_raw)
        ts_final = tf.image.resize(ts_tensor, [IMG_WIDTH, IMG_HEGHT])
        ts_final = ts_final / 255.0
        ts_final = (np.expand_dims(ts_final,0))
        ts_all.append(ts_final)
    # print (ts_all)
    for index in range(0,len(ts_all)):
        predict = prb_mdl.predict(ts_all[index])
        for i in range(0,2):
            print (ts_img_path[index]," predict:")
            print (class_names[i],"probility:=>%lf"%predict[0][i])
        print (" ")

def Test_Single(img_dir,prb_mdl,class_names,real):
    img_raw = tf.io.read_file(img_dir)
    img_tensor = tf.image.decode_image(img_raw)
    img_final = tf.image.resize(img_tensor, [IMG_WIDTH, IMG_HEGHT])
    img_final = img_final / 255.0
    img_final = (np.expand_dims(img_final,0))
    max_prb = 0
    max_index = 0
    predict = prb_mdl.predict(img_final)
    for index in range(0,len(class_names)):
        if (max_prb <= predict[0][index]):
            max_prb = predict[0][index]
            max_index = index
    print (img_dir,"is predicted as ",class_names[max_index],"prb is %lf"%predict[0][max_index])
    print (img_dir,"read is ",real)

if __name__ == "__main__":
    print ("Entry Main Function...")
    VERSION()
    GPU_SET()
    ds,labels_names,count = Create_Ds("/home/xny/tensorflow_workdir/data",IMG_BATCH)
    model = Create_Neural_Network(ds,IMG_BATCH,labels_names)
    model.fit(ds,epochs=2,steps_per_epoch=(count/IMG_BATCH))
    prb_mdl = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
    Test_Img("./ts",prb_mdl,labels_names)
    # Test_Single("./ts/car.jpg",prb_mdl,labels_names,"car")