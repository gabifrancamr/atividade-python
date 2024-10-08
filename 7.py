# Segmentação de Imagens: Limiarização, Watershed Algorithm.

import cv2
import numpy as np

imagem = cv2.imread('./img/3.webp', 0)

# Limiarização simples
_, imagem_binaria = cv2.threshold(imagem, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Limiarização', imagem_binaria)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Watershed Algorithm
imagem_rgb = cv2.imread('./img/3.webp')
imagem_cinza = cv2.cvtColor(imagem_rgb, cv2.COLOR_BGR2GRAY)
_, imagem_bin = cv2.threshold(imagem_cinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Remover ruído
kernel = np.ones((3, 3), np.uint8)
abertura = cv2.morphologyEx(imagem_bin, cv2.MORPH_OPEN, kernel, iterations=2)

# Obter fundo
fundo = cv2.dilate(abertura, kernel, iterations=3)

# Obter o primeiro plano
dist_transform = cv2.distanceTransform(abertura, cv2.DIST_L2, 5)
_, primeira_linha = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Watershed
primeira_linha = np.uint8(primeira_linha)
desconhecido = cv2.subtract(fundo, primeira_linha)
ret, marcadores = cv2.connectedComponents(primeira_linha)
marcadores = marcadores + 1
marcadores[desconhecido == 255] = 0
marcadores = cv2.watershed(imagem_rgb, marcadores)
imagem_rgb[marcadores == -1] = [255, 0, 0]

cv2.imshow('Watershed', imagem_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
