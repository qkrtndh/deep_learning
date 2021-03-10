import tensorflow as tf
import pandas as pd

(독립, 종속), _ = tf.keras.datasets.mnist.load_data()
print(독립.shape,종속.shape)
독립 =  독립.reshape(60000,28,28,1)
종속 = pd.get_dummies(종속)
print(독립.shape,종속.shape)

X = tf.keras.layers.Input(shape=[28,28,1])
H = tf.keras.layers.Conv2D(3,kernel_size=5,activation='swish')(X)#convolution layer
H = tf.keras.layers.MaxPool2D()(H)#pooling layer
H = tf.keras.layers.Conv2D(6,kernel_size=5,activation='swish')(H)#convolution layer
H = tf.keras.layers.MaxPool2D()(H)#pooling layer

H = tf.keras.layers.Flatten()(H)#flatten layer
H = tf.keras.layers.Dense(84,activation='swish')(H)#flatten layer
Y = tf.keras.layers.Dense(10,activation='softmax')(H)#flatten layer
model = tf.keras.models.Model(X,Y)
model.compile(loss='categorical_crossentropy',metrics='accuracy')

model.summary()

model.fit(독립,종속,epochs=10)