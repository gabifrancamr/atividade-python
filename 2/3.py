# Pré-processamento de Imagens: Conversão de Cores, Redimensionamento, Equalização de Histograma.

#EQUALIZAÇÃO DE HISTOGRAMA

import cv2
import matplotlib.pyplot as plt

imagem = cv2.imread('./img/1.jpg')

#convertendo imagem para cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

#equalização de histograma
imagem_equalizada = cv2.equalizeHist(imagem_cinza)
plt.imshow(imagem_equalizada, cmap='gray')
plt.title("Imagem equalizada")
plt.show()