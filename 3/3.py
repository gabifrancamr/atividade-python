# Aplicação de Filtros: Desfoque (Blur), 
# Detecção de Bordas (Canny, Sobel), Filtro Laplaciano.

#FILTRO LAPLACIANO

import cv2
import matplotlib.pyplot as plt

imagem = cv2.imread('./img/3.webp')
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

laplaciano = cv2.Laplacian(imagem_cinza, cv2.CV_64F)
laplaciano = cv2.convertScaleAbs(laplaciano)  # Converte a imagem para 8-bits para exibição

cv2.imshow('Filtro Laplaciano', laplaciano)
cv2.waitKey(0)
cv2.destroyAllWindows()
