import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#Fashiob Mnist
fashion_mnist = keras.datasets.fashion_mnist
(tr_image,tr_label),(ts_image,ts_label) = fashion_mnist.load_data()

#check the format of train image
# print (tr_image.shape)
# print (len(tr_label))
# print (tr_label)

# print (ts_image.shape)
# print (len(ts_label))
# print (ts_label)

# plt.figure()
# plt.imshow(tr_image[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

tr_image = tr_image / 255.0
ts_image = ts_image / 255.0

# plt.figure(figsize=(10,10))
# for i in range(0,25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(tr_image[i])
#     plt.xlabel(class_names[tr_label[i]])
# plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(tr_image,tr_label,epochs=10)

test_loss, test_acc = model.evaluate(ts_image, ts_label, verbose=2)

print ("测试图片分类的准确率:=>",test_acc)

prb_mdl = tf.keras.Sequential([model,tf.keras.layers.Softmax()])

# predict = prb_mdl.predict(ts_image) 

# tmps = 0
# ct = 0
# for i in range(0,10):
#     if (tmps < predict[0][i]):
#         tmps = predict[0][i]
#         ct = i
# print ("第一张测试图片的预测结果是:=>",class_names[ct])
# print ("第一张图片的标签真实值是:=>",class_names[ts_label[0]])

tmp_img = ts_image[2]
tmp_img = (np.expand_dims(tmp_img,0))

predict = prb_mdl.predict(tmp_img)
tmps = 0
ct = 0
for i in range(0,10):
    if (tmps < predict[0][i]):
        tmps = predict[0][i]
        ct = i
print ("单张预测结果是:=>",class_names[ct])
print ("单张图片的标签真实值是:=>",class_names[ts_label[2]])
