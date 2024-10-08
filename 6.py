# Operações Morfológicas: Erosão, Dilatação, Segmentação de Objetos.

import cv2
import numpy as np

# Carregar a imagem em escala de cinza
imagem = cv2.imread('./img/3.webp', 0)

# Aplicar limiarização para binarizar a imagem
_, imagem_binaria = cv2.threshold(imagem, 127, 255, cv2.THRESH_BINARY)

# Definir o kernel para operações morfológicas
kernel = np.ones((5, 5), np.uint8)

# 1. Erosão
erosao = cv2.erode(imagem_binaria, kernel, iterations=1)
cv2.imshow('Erosão', erosao)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. Dilatação
dilatacao = cv2.dilate(imagem_binaria, kernel, iterations=1)
cv2.imshow('Dilatação', dilatacao)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. Segmentação de Objetos - Usando Abertura (Erosão seguida de Dilatação)
abertura = cv2.morphologyEx(imagem_binaria, cv2.MORPH_OPEN, kernel)
cv2.imshow('Segmentação de Objetos (Abertura)', abertura)
cv2.waitKey(0)
cv2.destroyAllWindows()
