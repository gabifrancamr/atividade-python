# Operações Morfológicas: Erosão, Dilatação, Segmentação de Objetos.

import cv2
import numpy as np

# O segundo parâmetro (0) indica que a imagem será carregada em escala de cinza.
imagem = cv2.imread('./img/3.webp', 0)

# Aplicar limiarização para binarizar a imagem
# cv2.threshold converte a imagem em uma imagem binária.
# O primeiro argumento é a imagem de entrada.
# O segundo argumento (127) é o valor do limiar: pixels com valor acima de 127 se tornam 255 (branco), 
# e pixels com valor abaixo de 127 se tornam 0 (preto).
# O terceiro argumento (255) é o valor que será atribuído aos pixels acima do limiar.
# O quarto argumento (cv2.THRESH_BINARY) especifica que a operação de limiarização será binária.
_, imagem_binaria = cv2.threshold(imagem, 127, 255, cv2.THRESH_BINARY)

# Definir o kernel para operações morfológicas
# O kernel é uma matriz que será usada nas operações morfológicas (como dilatação e erosão).
# np.ones cria uma matriz de 5x5 preenchida com 1s (pixels brancos).
# np.uint8 especifica o tipo de dados da matriz, que neste caso é 8 bits sem sinal.
kernel = np.ones((5, 5), np.uint8)

# 1. Erosão
# A erosão é uma operação morfológica que remove pixels de bordas de objetos na imagem.
# cv2.erode recebe a imagem binária de entrada, o kernel (matriz de 5x5) e o número de iterações (1).
# Uma iteração significa que a erosão será aplicada uma vez.
erosao = cv2.erode(imagem_binaria, kernel, iterations=1)
cv2.imshow('Erosão', erosao)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. Dilatação
# A dilatação é uma operação morfológica que adiciona pixels às bordas dos objetos na imagem.
# cv2.dilate recebe a imagem binária de entrada, o kernel (matriz de 5x5) e o número de iterações (1).
# Assim como na erosão, uma iteração significa que a dilatação será aplicada uma vez.
dilatacao = cv2.dilate(imagem_binaria, kernel, iterations=1)
cv2.imshow('Dilatação', dilatacao)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. Segmentação de Objetos - Usando Abertura (Erosão seguida de Dilatação)
# A abertura é uma operação morfológica que é usada para remover ruídos pequenos de uma imagem,
# que é feita aplicando erosão seguida de dilatação.
# cv2.morphologyEx recebe a imagem binária de entrada, o tipo de operação (cv2.MORPH_OPEN) e o kernel (matriz de 5x5).
abertura = cv2.morphologyEx(imagem_binaria, cv2.MORPH_OPEN, kernel)
cv2.imshow('Segmentação de Objetos (Abertura)', abertura)
cv2.waitKey(0)
cv2.destroyAllWindows()
