#!/home/xny/anaconda3/envs/tf/bin/python3.7
# coding=utf-8
from numpy.lib.shape_base import expand_dims
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from pid import Pid
import os
import sys
import pathlib
import random

def Cap_Sensor_Init(cap_id):
    cap = cv.VideoCapture(cap_id)
    if not cap.isOpened():
        print ("摄像头打开失败.")
        return -1
    else:
        print ("摄像头设置成功.")
        return cap

def Cap_Sensor_Frame(cap):
    return cap.read()

def Gray_A_Bin_Deal(img,w,h):
    gray = 0
    for y in range(0,h):
        for x in range(0,w):
            gray = (int(img[y,x][0]) + int(img[y,x][1]) + int(img[y,x][2]))/3
            if (gray>127):
                gray = 255
            else:
                gray = 0
            img[y,x] = [gray,gray,gray]

# CNN FUNCTIONS
BATCH_SIZE = 32
IMG_W = 227
IMG_H = 227

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # base下默认使用cpu
AUTOTUNE = tf.data.experimental.AUTOTUNE  # 最优化的线程数

def Data_PreProcess():
    data_route = input ("请输入训练数据所在目录:")
    data_local = pathlib.Path(data_route)
    all_image_path = [str(sth) for sth in data_local.glob('*/*')]
    random.shuffle(all_image_path)
    # print ('(',all_image_path[:10],'...',')')
    class_name = [str(sth).split('/')[3] for sth in data_local.glob('*/') if sth.is_dir()]
    # print (class_name)
    label_to_index = dict( (name,index) for index,name in enumerate(class_name))
    # print (label_to_index)
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] \
                                           for path in all_image_path]
    # print ('(',all_image_labels[:10],'...',')')
    return len(all_image_path),all_image_path,all_image_labels,class_name,label_to_index
########## @ret: 图片张数       @ret: 图片路径  @ret: 图片标签     @ret: 图片类别  @ret: 图片类别索引 ##########

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

# Hex 2 Dec
def Hex_2_Dec(num):
    if num == 'a':
        return 10
    elif num == 'b':
        return 11
    elif num == 'c':
        return 12
    elif num == 'd':
        return 13
    elif num == 'e':
        return 14
    elif num == 'f':
        return 15
    else:
        return int(num)        

def Mnist_PreProcess(images_dir,labels_dir,nums):
     # 读图片数据
    all_images_bytes = tf.io.read_file(images_dir)
    all_images_hex = all_images_bytes.numpy().hex()
    # 十六进制数据 每一位4bit 省去头部32个数据
    offsets = 32
    pix_index = 0
    all_images_tensor = []
    for ct in range(0,nums):
        images_data = []
        for y in range(0,28):
            line_data = []
            for x in range(0,28):
                gray = Hex_2_Dec(all_images_hex[offsets+pix_index])*16 + Hex_2_Dec(all_images_hex[offsets+pix_index+1])
                pix = [gray,gray,gray] # 扩充成三通道
                line_data.append(pix)
                pix_index += 2
            images_data.append(line_data)
        all_images_tensor.append(images_data)
    all_images_tensor = np.array(all_images_tensor)
        
    
    # 读标签数据
    all_labels_bytes = tf.io.read_file(labels_dir)
    all_labels_hex = all_labels_bytes.numpy().hex()
    # 十六进制数据 每一位4bit 省去头部16个数据
    offsets = 16
    label_index = 0
    all_labels_tensor = []
    for ct in range(0,nums):
        all_labels_tensor.append(int(all_labels_hex[offsets+label_index+1]))
        label_index += 2
    all_labels_tensor = np.array(all_labels_tensor)
    return all_images_tensor,all_labels_tensor

# Minst Predict
def Minst_Predict(s_index,nums,ts_images_tensor,ts_labels_tensor,prb_mdl,ts_all):
    index = s_index
    while (index<s_index + nums):
        predict = prb_mdl.predict(ts_all[index])
        plt.subplot(2,nums,index-s_index+1)
        # cv.resize 无法处理 int 型数据
        img = cv.resize(ts_images_tensor[index].astype(np.float32),(96,96),interpolation=cv.INTER_LINEAR)
        plt.imshow(img.astype(np.uint8))
        title_name = 'Num in Pic:' + str(ts_labels_tensor[index])
        plt.title(title_name)
        
        plt.subplot(2,nums,index-s_index+nums+1)
        x = np.array(['zero','one','two','three','four','five','six','siven','eight','nine'])
        plt.bar(x,height=[predict[0][i]*100 for i in range(0,10)])
        plt.title('predict:')
        index += 1
    plt.show()

# Main Enter
if __name__ == "__main__":
    print (cv.__version__)
    print (tf.__version__)
    
# OpenCV Test
    # cap = Cap_Sensor_Init(0)
    # if (cap == -1):
    #     exit()

    # while (True):
    #     ret,img = Cap_Sensor_Frame(cap)
    #     if  ret == True:
    #         cv.imshow("cam",img)
    #         cv.waitKey(33)

# MatPlot Test
    # img = cv.imread("./env.jpg")
    # cv.cvtColor(img,cv.COLOR_RGB2BGR,img)
    # plt.imshow(img)
    # plt.show()
    
    # x_dst = [0,1,2,3,4,5,6,7,8,9]
    # y_dst = [3,7,8,9,2,5,4,6,3,7]
    # x_np_dst = np.array(x_dst)
    # y_np_dst = np.array(y_dst)
    # plt.plot (x_np_dst,y_np_dst)
    # plt.show()
    
# Pid & MatPlotLib Test
    # exp = [8.94]
    # Y   = [0]
    # X   = [0]
    # T   = 0.05
    # count = 0
    #              # kp  ki  kd
    # new_pid = Pid(0.12 ,0.12 ,0.01 ,exp ,Y ,X ,T ,False ,[-100,100])
    
    # axis_x = []
    # axis_y = []
    # axis_ey = []
    # fd = open('./pid_datasheet.txt',mode = 'w+')
    
    # while (True):
    #     Y[0] = -4*X[0] + 7.7
    #     ret = new_pid.Pid_Compute()
    #     if ret == True:
    #         print ("Y:[%0.2f]\t"%Y[0],end='')
    #         print ("X:[%0.2f]\t"%X[0],end='')
    #         print ("Exp:[%0.2f]\t"%exp[0])
    #         axis_x.append(count * T)
    #         axis_y.append(Y[0])
    #         axis_ey.append(exp[0])
    #         count += 1
    #         buf_y = "Y:[%0.2f]\t"%Y[0]
    #         buf_x = "X:[%0.2f]\t"%X[0]
    #         buf = buf_y + buf_x + "\r\n"
    #         fd.write(buf)
    #     if count == 50:
    #         break
    # line_1 = plt.plot(axis_x,axis_y)
    # plt.setp(line_1,'c','r','lw',2.0)
    # line_2 = plt.plot(axis_x,axis_ey)
    # plt.setp(line_2,'c','b','lw',2.0)
    # plt.show()
    # fd.close()
    
# CCN Test (Tensorflow Framework)
    # nums,all_images_path,all_images_labels,class_name,label_dict = Data_PreProcess()
    # ds = tf.data.Dataset.from_tensor_slices((all_images_path[:1000], all_images_labels[:1000]))
    # image_label_ds = ds.map(load_and_preprocess_from_path_label)
    # ds = image_label_ds.apply(
    # tf.data.experimental.shuffle_and_repeat(buffer_size=2000))
    # ds = ds.batch(BATCH_SIZE)
    # ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    # model = models.Sequential()
    # model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(IMG_W, IMG_H, 3)))
    # model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))
    # model.add(layers.Conv2D(256, (5, 5), padding='same', activation='relu'))
    # model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))
    # model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))
    # model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))
    # model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    # model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(4096,kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    # model.add(layers.Dense(4096,kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    # model.add(layers.Dense(1000,kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    # model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    # model.add(layers.Dense(30, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    # model.add(layers.Dense(len(class_name)))
    # print (model.summary())

    # model.compile(optimizer='adam',
    #           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #           metrics=['accuracy'])
    # history = model.fit(ds, epochs=3,steps_per_epoch=int(1000/BATCH_SIZE))
    # prb_mdl = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
    # Test_Img("./ts",prb_mdl,class_name)
    
# tensor & numpy test
    # img_by_tf = tf.io.read_file("./ts/cat.4001.jpg")
    # img_by_tf = tf.image.decode_image(img_by_tf,channels=3)
    # img_by_cv = cv.imread("./ts/cat.4001.jpg")
    
    # print (img_by_tf)
    # print (img_by_cv)
    # numpy 2 tensor
    # img_by_cv_cvrt_tf = tf.image.resize(img_by_cv,[300,300])
    # print (img_by_cv_cvrt_tf)
    
    # plt.subplot(121)
    # plt.imshow(img_by_tf.numpy().astype(np.uint8))
    # plt.title('By TF')
    # plt.subplot(122)
    # plt.imshow(img_by_cv.astype(np.uint8))
    # plt.title('By CV')
    # plt.show() 
    
# LeNet Test
    # all_images_tensor,all_labels_tensor = Mnist_PreProcess('./datasets/mnist/train-images-idx3-ubyte',\
    #    './datasets/mnist/train-labels-idx1-ubyte', 10000) # numpy
    # ts_images_tensor,ts_labels_tensor = Mnist_PreProcess('./datasets/mnist/t10k-images-idx3-ubyte',\
    #    './datasets/mnist/t10k-labels-idx1-ubyte', 1000)
    # show samples
    # for index in range(0,9):
    #     plt.subplot(3,3,index+1)
    #     plt.imshow(all_images_tensor[index])
    #     plt.title(str(all_labels_tensor[index]))
    #     plt.axis('off')
    # plt.show()
    
    # all_images_tensor = tf.image.resize(all_images_tensor,[32,32]) #转成tensor
    # all_images_tensor /= 255.0
    
    # ts_all = []
    # for item in ts_images_tensor:
    #     tmps = tf.image.resize(item,[32,32]) #转成tensor
    #     tmps /= 255.0
    #     tmps = (np.expand_dims(tmps.numpy(),0))
    #     ts_all.append(tmps)
    
    # dataset
    # ds = tf.data.Dataset.from_tensor_slices((all_images_tensor, all_labels_tensor))
    # ds = ds.apply(
    # tf.data.experimental.shuffle_and_repeat(buffer_size=10000))
    # ds = ds.batch(BATCH_SIZE)
    # ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    # model = models.Sequential()
    # model.add(layers.Conv2D(6, (5, 5), strides=(1, 1), padding='valid', activation='relu', input_shape=(32, 32, 3)))
    # model.add(layers.AveragePooling2D((2, 2), strides=(2, 2)))
    # model.add(layers.Conv2D(16, (5, 5), strides=(1, 1), padding='valid', activation='relu'))
    # model.add(layers.AveragePooling2D((2, 2), strides=(2, 2)))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(120,kernel_regularizer=regularizers.l2(0.001)))
    # model.add(layers.Dense(84,kernel_regularizer=regularizers.l2(0.001)))
    # model.add(layers.Dense(10))
    # print (model.summary())
    
    # model.compile(optimizer='adam',
    #           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #           metrics=['accuracy'])
    # model.fit( all_images_tensor, all_labels_tensor, batch_size = 32, epochs = 3)
    # 保存模型
    # tf.keras.models.save_model(model,'./minst')
    # 加载模型
    model = tf.keras.models.load_model('./minst')
    
    # dataset load
    # model.compile(optimizer='adam',
    #           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #           metrics=['accuracy'])
    # model.fit( ds, epochs = 3, steps_per_epoch=int(10000/BATCH_SIZE))
    
    # 预测模型
    prb_mdl = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
    # Minst_Predict(215,5,ts_images_tensor,ts_labels_tensor,prb_mdl,ts_all)
    
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print ('camera open failed...')
        exit(-1)
    plt.ion()
    while True:
        ret,img = cap.read()
        if ret == True:
            # img 为numpy array
            # resize
            img = cv.resize(img,(32,32),interpolation=cv.INTER_LINEAR)
            # gray
            img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
            Gray_A_Bin_Deal(img,img.shape[1],img.shape[0])
            tf_img = tf.image.resize(img,[ 32, 32])
            tf_tensor = tf_img/255.0
            tf_tensor = (np.expand_dims(tf_tensor.numpy(),0))
            predict = prb_mdl.predict(tf_tensor)
            max = 0
            num = -1
            for p in range(0,9):
                if max < predict[0][p]:
                    max = predict[0][p]
                    num = p
            if not num == -1:
                print ("Find Num :[%d]"%num)
            # cv.imshow("cam-gray",img)
            # cv.waitKey(10)
            plt.imshow(tf_img.numpy().astype(np.uint8))
            plt.show()
            plt.pause(0.1)
            plt.clf()
    