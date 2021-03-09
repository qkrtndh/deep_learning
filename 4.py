import pandas as pd
import tensorflow as tf

파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
아이리스 = pd.read_csv(파일경로)
print(아이리스.columns)

#원 핫 인코딩
아이리스 = pd.get_dummies(아이리스)
print(아이리스.columns)

독립 = 아이리스[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
종속 = 아이리스[['품종_setosa', '품종_versicolor',
       '품종_virginica']]
print(독립.shape,종속.shape)

X = tf.keras.layers.Input(shape=[4])
Y = tf.keras.layers.Dense(3,activation='softmax')(X)
model = tf.keras.models.Model(X,Y)
model.compile(loss='categorical_crossentropy',metrics='accuracy')

model.fit(독립,종속,epochs=10)

model.predict(독립[-5:])

print(종속[-5:])
model.get_weights()