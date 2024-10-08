# Pré-processamento de Imagens: Conversão de Cores, Redimensionamento, Equalização de Histograma.

#REDIMENSIONAMENTO

import cv2
import matplotlib.pyplot as plt

imagem = cv2.imread('./img/1.jpg')

#redimensionamento da imagem
imagem_redimensionada = cv2.resize(imagem, (400, 400))
imagem_rgb = cv2.cvtColor(imagem_redimensionada, cv2.COLOR_BGR2RGB)
plt.imshow(imagem_rgb)
plt.title("Imagem redimensionada")
plt.show()

