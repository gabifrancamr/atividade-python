#  Transformações Geométricas: Rotação, Translação, Transformação de Perspectiva (Homografia), Correção de Distorção.

import cv2
import numpy as np

# Carregar a imagem
imagem = cv2.imread('./img/1.jpg')
altura, largura = imagem.shape[:2] # Obtém a altura e a largura da imagem

imagem_redimensionada = cv2.resize(imagem, (500, 500))

# 1. Rotação
def rotacionar_imagem(imagem, angulo):
    centro = (largura // 5, altura // 5)  # Calcula o centro da imagem para rotação
    # O centro é definido como 1/5 da largura e altura da imagem

    matriz_rotacao = cv2.getRotationMatrix2D(centro, angulo, 1.0)  # Cria a matriz de rotação com o centro especificado, ângulo e fator de escala. O fator de escala (1.0) indica que a imagem não será redimensionada durante a rotação

    # Aplica a transformação de rotação à imagem
    imagem_rotacionada = cv2.warpAffine(imagem, matriz_rotacao, (largura, altura))

    return imagem_rotacionada

imagem_rotacionada = rotacionar_imagem(imagem_redimensionada, 60)
cv2.imshow('Imagem Rotacionada', imagem_rotacionada)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. Translação
def transladar_imagem(imagem, deslocamento_x, deslocamento_y):

    # Cria a matriz de translação usando um array 2D
    # A matriz é definida como:
    # [[1, 0, deslocamento_x],  # Coluna 1: Fator de escala em x (1), sem rotação (0), e deslocamento em x
    #  [0, 1, deslocamento_y]]  # Coluna 2: Fator de escala em y (1), sem rotação (0), e deslocamento em y
    # 'deslocamento_x' e 'deslocamento_y' definem quantos pixels a imagem será movida
    matriz_translacao = np.float32([[1, 0, deslocamento_x], [0, 1, deslocamento_y]])

    # Aplica a transformação de translação à imagem
    # cv2.warpAffine recebe a imagem original, a matriz de translação e as dimensões da imagem (largura, altura)
    imagem_transladada = cv2.warpAffine(imagem, matriz_translacao, (largura, altura))

    return imagem_transladada

imagem_transladada = transladar_imagem(imagem_redimensionada, 500, 50)
cv2.imshow('Imagem Transladada', imagem_transladada)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. Transformação de Perspectiva (Homografia)
def transformar_perspectiva(imagem):
    # Define os pontos originais na imagem
    # Cada ponto é representado por suas coordenadas (x, y) em uma lista de quatro pontos
    # [50, 50]: canto superior esquerdo
    # [200, 50]: canto superior direito
    # [50, 200]: canto inferior esquerdo
    # [200, 200]: canto inferior direito
    pontos_originais = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])  

    # Define os pontos finais para a transformação de perspectiva
    # Esses pontos correspondem a onde os pontos originais devem ser mapeados
    # [10, 100]: novo ponto para o canto superior esquerdo
    # [180, 50]: novo ponto para o canto superior direito
    # [100, 250]: novo ponto para o canto inferior esquerdo
    # [200, 200]: mantém o canto inferior direito inalterado
    pontos_finais = np.float32([[10, 100], [180, 50], [100, 250], [200, 200]])  

    # Cria a matriz de homografia que descreve a transformação de perspectiva
    # A matriz é calculada usando os pontos originais e finais
    matriz_homografia = cv2.getPerspectiveTransform(pontos_originais, pontos_finais)

    # Aplica a transformação de perspectiva à imagem
    # cv2.warpPerspective recebe a imagem original, a matriz de homografia e as dimensões da imagem (largura, altura)
    imagem_perspectiva = cv2.warpPerspective(imagem, matriz_homografia, (largura, altura))

    return imagem_perspectiva

imagem_perspectiva = transformar_perspectiva(imagem_redimensionada)
cv2.imshow('Transformação de Perspectiva', imagem_perspectiva)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4. Correção de Distorção (simulação simples)
def corrigir_distorcao(imagem):
    # Simulação de matriz de calibração da câmera
    # A matriz é composta pelos seguintes elementos:
    # fx: Fator de escala em x (1000) - afeta a largura da imagem
    # fy: Fator de escala em y (1000) - afeta a altura da imagem
    # cx: Coordenada x do centro da imagem (largura // 2) - coloca o centro no meio da imagem
    # cy: Coordenada y do centro da imagem (altura // 2) - coloca o centro no meio da imagem
    # A matriz é um array 3x3 que é usado para modelar a câmera
    matriz_camera = np.array([[1000, 0, largura // 2], [0, 1000, altura // 2], [0, 0, 1]])

    # Coeficientes de distorção da lente
    # Exemplo simplificado com:
    # -0.2: coeficiente de distorção radial que corrige a distorção de barril
    # 0.1: coeficiente de distorção que pode corrigir a distorção de pinça
    # Os dois zeros são para outros coeficientes que não estão sendo usados
    coef_distorcao = np.array([[-0.2, 0.1, 0, 0]])  

    # Aplica a correção de distorção à imagem
    # cv2.undistort recebe a imagem original, a matriz de calibração da câmera e os coeficientes de distorção
    imagem_corrigida = cv2.undistort(imagem, matriz_camera, coef_distorcao)
    return imagem_corrigida
    

imagem_corrigida = corrigir_distorcao(imagem_redimensionada)
cv2.imshow('Correção de Distorção', imagem_corrigida)
cv2.waitKey(0)
cv2.destroyAllWindows()
