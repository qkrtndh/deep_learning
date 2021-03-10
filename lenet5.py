import tensorflow as tf
import pandas as pd

(독립, 종속), _ = tf.keras.datasets.mnist.load_data()
print(독립.shape,종속.shape)
독립 =  독립.reshape(60000,28,28,1)
종속 = pd.get_dummies(종속)
print(독립.shape,종속.shape)

X = tf.keras.layers.Input(shape=[28,28,1])
H = tf.keras.layers.Conv2D(6,kernel_size=5,padding='same',activation='swish')(X)#convolution layer
H = tf.keras.layers.MaxPool2D()(H)#pooling layer
H = tf.keras.layers.Conv2D(16,kernel_size=5,activation='swish')(H)#convolution layer
H = tf.keras.layers.MaxPool2D()(H)#pooling layer

H = tf.keras.layers.Flatten()(H)#flatten layer
H = tf.keras.layers.Dense(120,activation='swish')(H)#flatten layer
H = tf.keras.layers.Dense(84,activation='swish')(H)#flatten layer
Y = tf.keras.layers.Dense(10,activation='softmax')(H)#flatten layer
model = tf.keras.models.Model(X,Y)
model.compile(loss='categorical_crossentropy',metrics='accuracy')

model.fit(독립,종속,epochs=10)
##################
#cifar10
(독립, 종속), _ = tf.keras.datasets.cifar10.load_data()
print(독립.shape,종속.shape)
#독립변수가 이미 4차원이기 때문에 reshape불필요
#mnist는 종속변수가 1차원이라 상관 없었지만, 이번에는 2차원이기 때문에 1차원으로 reshape 해주어야 함
종속 = pd.get_dummies(종속.reshape(50000))
print(독립.shape,종속.shape)

X = tf.keras.layers.Input(shape=[32,32,3])
H = tf.keras.layers.Conv2D(6,kernel_size=5,activation='swish')(X)#convolution layer
H = tf.keras.layers.MaxPool2D()(H)#pooling layer
H = tf.keras.layers.Conv2D(16,kernel_size=5,activation='swish')(H)#convolution layer
H = tf.keras.layers.MaxPool2D()(H)#pooling layer

H = tf.keras.layers.Flatten()(H)#flatten layer
H = tf.keras.layers.Dense(120,activation='swish')(H)#flatten layer
H = tf.keras.layers.Dense(84,activation='swish')(H)#flatten layer
Y = tf.keras.layers.Dense(10,activation='softmax')(H)#flatten layer
model = tf.keras.models.Model(X,Y)
model.compile(loss='categorical_crossentropy',metrics='accuracy')

model.fit(독립,종속,epochs=10)

model.summary()