# Aplicação de Filtros: Desfoque (Blur), 
# Detecção de Bordas (Canny, Sobel), Filtro Laplaciano.

#DETECÇÃO DE BORDAS

import cv2
import matplotlib.pyplot as plt

imagem = cv2.imread('./img/1.jpg')

imagem2 = cv2.imread('./img/3.webp')
imagem_cinza = cv2.cvtColor(imagem2, cv2.COLOR_BGR2GRAY)

bordas = cv2.Canny(imagem_cinza, 100, 200)

cv2.imshow('Bordas', bordas)
cv2.waitKey(0)
cv2.destroyAllWindows()
