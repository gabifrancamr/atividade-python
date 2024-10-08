#  Transformações Geométricas: Rotação, Translação, Transformação de Perspectiva (Homografia), Correção de Distorção.

import cv2
import numpy as np

# Carregar a imagem
imagem = cv2.imread('./img/1.jpg')
altura, largura = imagem.shape[:2]

imagem_redimensionada = cv2.resize(imagem, (500, 500))

# 1. Rotação
def rotacionar_imagem(imagem, angulo):
    centro = (largura // 5, altura // 5)  # Centro da imagem
    matriz_rotacao = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    imagem_rotacionada = cv2.warpAffine(imagem, matriz_rotacao, (largura, altura))
    return imagem_rotacionada

imagem_rotacionada = rotacionar_imagem(imagem_redimensionada, 60)
cv2.imshow('Imagem Rotacionada', imagem_rotacionada)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. Translação
def transladar_imagem(imagem, deslocamento_x, deslocamento_y):
    matriz_translacao = np.float32([[1, 0, deslocamento_x], [0, 1, deslocamento_y]])
    imagem_transladada = cv2.warpAffine(imagem, matriz_translacao, (largura, altura))
    return imagem_transladada

imagem_transladada = transladar_imagem(imagem_redimensionada, 500, 50)
cv2.imshow('Imagem Transladada', imagem_transladada)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. Transformação de Perspectiva (Homografia)
def transformar_perspectiva(imagem):
    pontos_originais = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])  # Pontos originais
    pontos_finais = np.float32([[10, 100], [180, 50], [100, 250], [200, 200]])  # Pontos desejados

    matriz_homografia = cv2.getPerspectiveTransform(pontos_originais, pontos_finais)
    imagem_perspectiva = cv2.warpPerspective(imagem, matriz_homografia, (largura, altura))
    return imagem_perspectiva

imagem_perspectiva = transformar_perspectiva(imagem_redimensionada)
cv2.imshow('Transformação de Perspectiva', imagem_perspectiva)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4. Correção de Distorção (simulação simples)
def corrigir_distorcao(imagem):
    # Simulação de matriz de calibração da câmera (fx, fy, cx, cy) e coeficientes de distorção
    matriz_camera = np.array([[1000, 0, largura // 2], [0, 1000, altura // 2], [0, 0, 1]])
    coef_distorcao = np.array([[-0.2, 0.1, 0, 0]])  # Exemplo simplificado de coeficiente

    imagem_corrigida = cv2.undistort(imagem, matriz_camera, coef_distorcao)
    return imagem_corrigida
    

imagem_corrigida = corrigir_distorcao(imagem_redimensionada)
cv2.imshow('Correção de Distorção', imagem_corrigida)
cv2.waitKey(0)
cv2.destroyAllWindows()
