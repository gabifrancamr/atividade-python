# Pré-processamento de Imagens: Conversão de Cores, Redimensionamento, Equalização de Histograma.

#CONVERSÃO DE CORES

import cv2
import matplotlib.pyplot as plt

imagem = cv2.imread('./img/1.jpg')

#convertendo imagem para cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
plt.imshow(imagem_cinza, cmap='gray')
plt.title("Imagem transformada em cinza")
plt.show()