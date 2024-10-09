# Aplicação de Filtros: Desfoque (Blur), 
# Detecção de Bordas (Canny, Sobel), Filtro Laplaciano.

#FILTRO LAPLACIANO

import cv2
import matplotlib.pyplot as plt

imagem = cv2.imread('./img/3.webp')
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Aplica o operador Laplaciano para detectar mudanças abruptas de intensidade (bordas) na imagem em escala de cinza
laplaciano = cv2.Laplacian(imagem_cinza, cv2.CV_64F) # Define o formato da imagem como ponto flutuante de 64 bits para permitir valores negativos e grandes

# Converte a imagem Laplaciana para uma escala de 8 bits (valores absolutos) para exibição
laplaciano = cv2.convertScaleAbs(laplaciano)  

cv2.imshow('Filtro Laplaciano', laplaciano)
cv2.waitKey(0)
cv2.destroyAllWindows()
