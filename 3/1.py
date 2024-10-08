# Aplicação de Filtros: Desfoque (Blur), 
# Detecção de Bordas (Canny, Sobel), Filtro Laplaciano.

#DESFOQUE

import cv2
import matplotlib.pyplot as plt

imagem = cv2.imread('./img/1.jpg')

imagem_redimensionada = cv2.resize(imagem, (400, 400))
desfoque = cv2.GaussianBlur(imagem_redimensionada, (15, 15), 0)
cv2.imshow('Imagem suavizada', desfoque)
cv2.waitKey(0)
cv2.destroyAllWindows()
