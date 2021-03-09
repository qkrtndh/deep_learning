import pandas as pd
import tensorflow as tf
#데이터 가져오기
파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
레모네이드 = pd.read_csv(파일경로)
레모네이드.head()

#독립, 종속
독립 = 레모네이드[['온도']]
종속 = 레모네이드[['판매량']]
print(독립.shape,종속.shape)

#모델 만들기
X = tf.keras.layers.Input(shape=[1])#독립데이터의 데이터 수, 인풋레이어를 만든다
Y = tf.keras.layers.Dense(1)(X)#종속변수 column 1개, 덴스 레이어
model = tf.keras.models.Model(X,Y)#종속변수와 독립변수를 넣어서 모델 생성
model.compile(loss='mse')#모델이 학습 할 방법

#모델 학습
model.fit(독립,종속,epochs=10)#model.fit(독립,종속,epochs=10000, verbose=0) 학습결과 표시 안함

#모델을 이용한다
model.predict(독립)

#정답 확인
종속

#모델 적용해서 예측하기
model.predict([[15]])