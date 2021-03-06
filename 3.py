import pandas as pd
import tensorflow as tf

파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
보스턴 = pd.read_csv(파일경로)
print(보스턴.columns)

독립 = 보스턴[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
       'ptratio', 'b', 'lstat']]
종속 = 보스턴[['medv']]
print(독립.shape, 종속.shape)

X=tf.keras.layers.Input(shape=[13])
H = tf.keras.layers.Dense(10)(X)
H = tf.keras.layers.Activation('swish')(H)
H = tf.keras.layers.BatchNormalization()(H)

H = tf.keras.layers.Dense(10)(H)
H = tf.keras.layers.Activation('swish')(H)
H = tf.keras.layers.BatchNormalization()(H)

H = tf.keras.layers.Dense(10)(H)
H = tf.keras.layers.Activation('swish')(H)
H = tf.keras.layers.BatchNormalization()(H)

Y = tf.keras.layers.Dense(1)(H)
model = tf.keras.models.Model(X,Y)
model.compile(loss='mse')

#모델에 히든레이어 됬는지 확인용
model.summary()

model.fit(독립,종속,epochs=10)

model.predict(독립[0:5])

종속[0:5]

#모델의 수식 확인
model.get_weights() 