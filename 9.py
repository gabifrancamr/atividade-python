# Análise de Imagens: Análise de Forma, Análise de Textura (Local Binary Patterns).

import cv2
import numpy as np

imagem1 = cv2.imread('./img/1.jpg')
imagem2 = cv2.imread('./img/2.png')

# Análise de Forma - Contornos
imagem_cinza = cv2.cvtColor(imagem1, cv2.COLOR_BGR2GRAY)
_, imagem_binaria = cv2.threshold(imagem_cinza, 127, 255, cv2.THRESH_BINARY)
contornos, _ = cv2.findContours(imagem_binaria, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(imagem1, contornos, -1, (0, 255, 0), 2)
cv2.imshow('Análise de Forma (Contornos)', imagem1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Análise de Textura - Local Binary Patterns (LBP)
def lbp_calculate(imagem):
    lbp = np.zeros_like(imagem)
    for i in range(1, imagem.shape[0] - 1):
        for j in range(1, imagem.shape[1] - 1):
            center = imagem[i, j]
            binary_str = ''
            binary_str += '1' if imagem[i-1, j-1] >= center else '0'
            binary_str += '1' if imagem[i-1, j] >= center else '0'
            binary_str += '1' if imagem[i-1, j+1] >= center else '0'
            binary_str += '1' if imagem[i, j+1] >= center else '0'
            binary_str += '1' if imagem[i+1, j+1] >= center else '0'
            binary_str += '1' if imagem[i+1, j] >= center else '0'
            binary_str += '1' if imagem[i+1, j-1] >= center else '0'
            binary_str += '1' if imagem[i, j-1] >= center else '0'
            lbp[i, j] = int(binary_str, 2)
    return lbp

imagem_lbp = lbp_calculate(imagem_cinza)
cv2.imshow('Análise de Textura (LBP)', imagem_lbp)
cv2.waitKey(0)
cv2.destroyAllWindows()
