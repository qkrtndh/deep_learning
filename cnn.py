import tensorflow as tf
import pandas as pd

#reshape 이용
(독립 ,종속), _ = tf.keras.datasets.mnist.load_data()
print(독립.shape,종속.shape)

#표의 형태로 바꾸는 과정
독립 = 독립.reshape(60000,784)
종속 = pd.get_dummies(종속)
#784열을 가진 독립변수, 10열을 가진 종속변수가 만들어진다
print(독립.shape,종속.shape)

#모델 생성
X = tf.keras.layers.Input(shape=[784])
H = tf.keras.layers.Dense(84, activation='swish')(X)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)
model = tf.keras.models.Model(X,Y)
model.compile(loss='categorical_crossentropy',metrics='accuracy')

#모델 학습
model.fit(독립,종속,epochs=10)

#모델 사용, 보기 편하게 출력
pred = model.predict(독립[0:5])
pd.DataFrame(pred).round(2)

종속[0:5]#정답 비교

###################################

#flatten 방식
(독립 ,종속), _ = tf.keras.datasets.mnist.load_data()
print(독립.shape,종속.shape)

#표의 형태로 바꾸는 과정
#독립 = 독립.reshape(60000,784)
종속 = pd.get_dummies(종속)
#784열을 가진 독립변수, 10열을 가진 종속변수가 만들어진다
print(독립.shape,종속.shape)

#모델 생성
X = tf.keras.layers.Input(shape=[28,28])
H = tf.keras.layers.Flatten()(X)
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)
model = tf.keras.models.Model(X,Y)
model.compile(loss='categorical_crossentropy',metrics='accuracy')

#모델 학습
model.fit(독립,종속,epochs=10)

#모델 사용, 보기 편하게 출력
pred = model.predict(독립[0:5])
pd.DataFrame(pred).round(2)

종속[0:5]#정답 비교