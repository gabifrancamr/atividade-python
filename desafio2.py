# Detecção de Anomalias: Detectar anomalias em imagens (objetos fora do padrão).

import cv2
import numpy as np

# Carregar a imagem padrão (normal) e a imagem para teste
imagem_normal = cv2.imread('./img/cachorro.jpg')
imagem_test = cv2.imread('./img/cachorro2.jpg')

# Converter as imagens para escala de cinza
imagem_normal_gray = cv2.cvtColor(imagem_normal, cv2.COLOR_BGR2GRAY)
imagem_test_gray = cv2.cvtColor(imagem_test, cv2.COLOR_BGR2GRAY)

# Calcular a diferença entre as imagens
diferenca = cv2.absdiff(imagem_normal_gray, imagem_test_gray)

# Aplicar um limiar para destacar as anomalias
_, limiar = cv2.threshold(diferenca, 30, 255, cv2.THRESH_BINARY)

# Exibir a imagem resultante
cv2.imshow('Detecção de Anomalias', limiar)
cv2.waitKey(0)
cv2.destroyAllWindows()
