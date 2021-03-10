import tensorflow as tf
import pandas as pd

#이미지 불러오기
(독립, 종속), _ = tf.keras.datasets.mnist.load_data()
print(독립.shape,종속.shape)

#필터에 들어갈 수 있도록 3차원으로 각 이미지 수정
독립 =  독립.reshape(60000,28,28,1)

#데이터 변환
종속 = pd.get_dummies(종속)
print(독립.shape,종속.shape)

X = tf.keras.layers.Input(shape=[28,28,1])#입력
H = tf.keras.layers.Conv2D(3,kernel_size=5,activation='swish')(X)#필터
H = tf.keras.layers.Conv2D(6,kernel_size=5,activation='swish')(H)#필터 총 6개의 채널 생성
H = tf.keras.layers.Flatten()(H)#이미지 데이터를 배열로 변환
H = tf.keras.layers.Dense(84,activation='swish')(H)#히든레이어 적용
Y = tf.keras.layers.Dense(10,activation='softmax')(H)#출력
model = tf.keras.models.Model(X,Y)
model.compile(loss='categorical_crossentropy',metrics='accuracy')

#학습
model.fit(독립,종속,epochs=10)

#테스트
pred=model.predict(독립[0:5])
pd.DataFrame(pred).round(2)

#결과비교
종속[0:5]


#모델 확인
model.summary()