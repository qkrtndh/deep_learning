import tensorflow as tf
#tensorlfow 에서 제공하는 샘플 이미지 다운로드 및 차원? 확인
(mnist_x,mnist_y), _ = tf.keras.datasets.mnist.load_data()
print(mnist_x.shape,mnist_y.shape)

(cifar_x,cifar_y), _ = tf.keras.datasets.cifar10.load_data()
print(cifar_x.shape,cifar_y.shape)

#이미지 확인
import matplotlib.pyplot as plt
print(mnist_y[0:10])
#흑맥이므로 cmap='gray' 명시할것
plt.imshow(mnist_x[0],cmap='gray')

print(cifar_y[0:10])
plt.imshow(cifar_x[0])

#차원 확인 word 파일에 상세내용
import numpy as np
d1 = np.array([1,2,3,4,5])
print(d1.shape)
d2 = np.array([d1,d1,d1,d1])
print(d2.shape)
d3 = np.array([d2,d2,d2])
print(d3.shape)
d4 = np.array([d3,d3])
print(d4.shape)