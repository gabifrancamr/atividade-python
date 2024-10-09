# Análise de Imagens: Análise de Forma, Análise de Textura (Local Binary Patterns).

import cv2
import numpy as np

imagem1 = cv2.imread('./img/1.jpg')
imagem2 = cv2.imread('./img/2.png')

# Análise de Forma - Contornos
# Converter a imagem para escala de cinza para facilitar a análise.
imagem_cinza = cv2.cvtColor(imagem1, cv2.COLOR_BGR2GRAY)

# Aplicar limiarização para binarizar a imagem.
# cv2.threshold transforma a imagem em uma imagem binária (preto e branco).
_, imagem_binaria = cv2.threshold(imagem_cinza, 127, 255, cv2.THRESH_BINARY)

# Encontrar contornos na imagem binária.
# cv2.findContours detecta contornos, que são os limites dos objetos na imagem.
contornos, _ = cv2.findContours(imagem_binaria, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Desenhar os contornos encontrados na imagem original.
# cv2.drawContours desenha contornos na imagem original (imagem1) com cor verde (0, 255, 0) e espessura 2.
cv2.drawContours(imagem1, contornos, -1, (0, 255, 0), 2)

cv2.imshow('Análise de Forma (Contornos)', imagem1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Análise de Textura - Local Binary Patterns (LBP)
def lbp_calculate(imagem):
    # Inicializar uma matriz LBP com zeros do mesmo tamanho da imagem.
    lbp = np.zeros_like(imagem)

    # Percorrer cada pixel da imagem, ignorando as bordas.
    for i in range(1, imagem.shape[0] - 1):
        for j in range(1, imagem.shape[1] - 1):
            center = imagem[i, j] # Valor do pixel central
            binary_str = '' # String binária para armazenar os valores

            # Comparar os pixels vizinhos com o pixel central e construir a string binária.
            binary_str += '1' if imagem[i-1, j-1] >= center else '0'  # Canto superior esquerdo
            binary_str += '1' if imagem[i-1, j] >= center else '0'    # Cima
            binary_str += '1' if imagem[i-1, j+1] >= center else '0'  # Canto superior direito
            binary_str += '1' if imagem[i, j+1] >= center else '0'    # Direita
            binary_str += '1' if imagem[i+1, j+1] >= center else '0'  # Canto inferior direito
            binary_str += '1' if imagem[i+1, j] >= center else '0'    # Baixo
            binary_str += '1' if imagem[i+1, j-1] >= center else '0'  # Canto inferior esquerdo
            binary_str += '1' if imagem[i, j-1] >= center else '0'    # Esquerda

            # Converter a string binária para um número inteiro e armazenar na matriz LBP.
            lbp[i, j] = int(binary_str, 2)
    return lbp

imagem_lbp = lbp_calculate(imagem_cinza)
cv2.imshow('Análise de Textura (LBP)', imagem_lbp)
cv2.waitKey(0)
cv2.destroyAllWindows()
