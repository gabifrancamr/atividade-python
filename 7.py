# Segmentação de Imagens: Limiarização, Watershed Algorithm.

import cv2
import numpy as np

# O segundo parâmetro (0) indica que a imagem será carregada em escala de cinza.
imagem = cv2.imread('./img/3.webp', 0)

# Limiarização simples
# cv2.threshold converte a imagem em uma imagem binária.
# O primeiro argumento é a imagem de entrada.
# O segundo argumento (127) é o valor do limiar: pixels com valor acima de 127 se tornam 255 (branco),
# e pixels com valor abaixo de 127 se tornam 0 (preto).
# O terceiro argumento (255) é o valor que será atribuído aos pixels acima do limiar.
# O quarto argumento (cv2.THRESH_BINARY) especifica que a operação de limiarização será binária.
_, imagem_binaria = cv2.threshold(imagem, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Limiarização', imagem_binaria)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Watershed Algorithm
imagem_rgb = cv2.imread('./img/3.webp')
imagem_cinza = cv2.cvtColor(imagem_rgb, cv2.COLOR_BGR2GRAY)

# Aplicar limiarização para criar uma imagem binária inversa
# cv2.threshold converte a imagem em uma imagem binária invertida usando Otsu.
# O primeiro argumento é a imagem de entrada em escala de cinza.
# O segundo argumento (0) é o valor do limiar. Aqui, Otsu calculará automaticamente o valor ideal.
# O terceiro argumento (255) é o valor que será atribuído aos pixels acima do limiar.
# O quarto argumento (cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) especifica que a operação será binária inversa,
# onde pixels acima do limiar se tornam 0 e abaixo se tornam 255.
_, imagem_bin = cv2.threshold(imagem_cinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Remover ruído
# Definindo um kernel 3x3 para operações morfológicas.
# np.ones cria uma matriz de 3x3 preenchida com 1s (pixels brancos).
kernel = np.ones((3, 3), np.uint8)

# Aplicar a operação de abertura para remover ruídos pequenos.
# A abertura é uma operação morfológica que realiza erosão seguida de dilatação.
# cv2.morphologyEx recebe a imagem binária de entrada, o tipo de operação (cv2.MORPH_OPEN),
# o kernel (matriz de 3x3) e o número de iterações (2).
abertura = cv2.morphologyEx(imagem_bin, cv2.MORPH_OPEN, kernel, iterations=2)

# Obter fundo
# A dilatação expande os objetos na imagem.
# cv2.dilate recebe a imagem da abertura, o kernel (matriz de 3x3) e o número de iterações (3).
fundo = cv2.dilate(abertura, kernel, iterations=3)

# Obter o primeiro plano
# A transformação de distância calcula a distância de cada pixel em relação ao pixel mais próximo de um objeto (ou seja, uma borda).
# cv2.distanceTransform recebe a imagem da abertura, o tipo de distância (cv2.DIST_L2) e um tamanho do kernel (5).
dist_transform = cv2.distanceTransform(abertura, cv2.DIST_L2, 5)

# Aplicar limiarização na transformação de distância para identificar o primeiro plano.
# O limiar (0.7 * dist_transform.max()) define um valor de corte para criar a imagem binária.
# Os pixels com valores acima deste limiar se tornam 255 (branco) e os abaixo se tornam 0 (preto).
_, primeira_linha = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Watershed
# Converte a imagem binária do primeiro plano para o tipo uint8.
primeira_linha = np.uint8(primeira_linha)

# O "desconhecido" é obtido subtraindo o primeiro plano do fundo.
# Isso resulta na área que não está coberta por objetos (ruído ou áreas vazias).
desconhecido = cv2.subtract(fundo, primeira_linha)

# Encontrar componentes conectados na imagem binária do primeiro plano.
# cv2.connectedComponents retorna o número de componentes e uma matriz de marcadores.
ret, marcadores = cv2.connectedComponents(primeira_linha)

# Adiciona 1 aos marcadores para evitar que o fundo seja considerado como parte de um objeto.
marcadores = marcadores + 1

# Marca a área desconhecida (ruído) como 0, para que não seja considerada na segmentação.
marcadores[desconhecido == 255] = 0

# Aplica o algoritmo Watershed, que é uma técnica de segmentação de imagem.
# cv2.watershed retorna os marcadores atualizados após a segmentação.
marcadores = cv2.watershed(imagem_rgb, marcadores)

# Marca os limites dos objetos detectados na imagem original em vermelho.
# Os marcadores com valor -1 representam os limites entre os objetos segmentados.
imagem_rgb[marcadores == -1] = [255, 0, 0]

cv2.imshow('Watershed', imagem_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
