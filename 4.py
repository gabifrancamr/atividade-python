# Detecção de Características: Detecção de Cantos (Harris, Shi-Tomasi),
# Detecção de Contornos,
# Pontos de Interesse (SIFT/SURF)

import cv2
import numpy as np

# Carregar a imagem
imagem = cv2.imread('./img/1.jpg')

# Redimensionar a imagem para 400x400
imagem_redimensionada = cv2.resize(imagem, (400, 400))

# Converter para escala de cinza
imagem_cinza = cv2.cvtColor(imagem_redimensionada, cv2.COLOR_BGR2GRAY)

# 1. Detecção de Cantos - Harris


def detectar_cantos_harris(imagem):
    # Converte a imagem para o formato de ponto flutuante de 32 bits, necessário para o detector Harris
    imagem_float = np.float32(imagem)
    # Aplica o detector de cantos de Harris (2 é o tamanho do bloco, 3 é o tamanho do filtro Sobel, 0.04 é o parâmetro de sensibilidade)
    cantos_harris = cv2.cornerHarris(imagem_float, 2, 3, 0.04)
    # Cria uma cópia da imagem redimensionada para marcar os cantos
    imagem_harris = imagem_redimensionada.copy()
    # Marca os pontos detectados como cantos (onde a resposta do Harris é maior que 1% do valor máximo) em vermelho
    imagem_harris[cantos_harris > 0.01 * cantos_harris.max()] = [0, 0, 255]
    return imagem_harris  # Retorna a imagem com os cantos marcados


harris_corners = detectar_cantos_harris(imagem_cinza)
cv2.imshow('Detecção de Cantos - Harris', harris_corners)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. Detecção de Cantos - Shi-Tomasi (Good Features to Track)


def detectar_cantos_shi_tomasi(imagem):
    # Detecta até 100 melhores cantos usando o algoritmo Shi-Tomasi (0.01 é o limite mínimo de qualidade, 10 é a distância mínima entre os cantos)
    cantos_shi_tomasi = cv2.goodFeaturesToTrack(imagem, 100, 0.01, 10)

    # Converte os cantos detectados para inteiros
    cantos_shi_tomasi = np.int0(cantos_shi_tomasi)

    # Cria uma cópia da imagem redimensionada para marcar os cantos
    imagem_shi_tomasi = imagem_redimensionada.copy()

    # Marca os cantos detectados como círculos verdes na imagem
    for canto in cantos_shi_tomasi:
        # Transforma o vetor de coordenadas (canto) em dois valores separados: x e y
        x, y = canto.ravel()

        cv2.circle(imagem_shi_tomasi, (x, y), 5, (0, 255, 0), -1)
        # Desenha um círculo verde nos cantos na imagem:
        # imagem_shi_tomasi: imagem onde o círculo será desenhado
        # (x, y): coordenadas do centro do círculo
        # 5: raio do círculo (tamanho)
        # (0, 255, 0): cor do círculo (verde, no formato BGR)
        # -1: espessura do círculo (-1 preenche o círculo)

    return imagem_shi_tomasi


shi_tomasi_corners = detectar_cantos_shi_tomasi(imagem_cinza)
cv2.imshow('Detecção de Cantos - Shi-Tomasi', shi_tomasi_corners)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. Detecção de Contornos


def detectar_contornos(imagem):
    # Detecta as bordas na imagem usando o algoritmo Canny (com limiares de 100 e 200)
    bordas = cv2.Canny(imagem, 100, 200)

    # Encontra os contornos a partir da imagem de bordas
    # cv2.RETR_EXTERNAL: Apenas os contornos externos são detectados
    # cv2.CHAIN_APPROX_SIMPLE: Simplifica os contornos, removendo pontos redundantes
    contornos, _ = cv2.findContours(
        bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Cria uma cópia da imagem redimensionada para desenhar os contornos
    imagem_contornos = imagem_redimensionada.copy()

    # Desenha os contornos detectados na imagem:
    # imagem_contornos: imagem onde os contornos serão desenhados
    # contornos: lista de contornos encontrados
    # -1: desenha todos os contornos
    # (255, 0, 0): cor dos contornos (azul, no formato BGR)
    # 2: espessura das linhas dos contornos
    cv2.drawContours(imagem_contornos, contornos, -1,
                     (255, 0, 0), 2)  
    
    return imagem_contornos


contours_image = detectar_contornos(imagem_cinza)
cv2.imshow('Detecção de Contornos', contours_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4. Detecção de Pontos de Interesse - SIFT


def detectar_pontos_interesse_sift(imagem):
    # Cria um objeto SIFT (Scale-Invariant Feature Transform) para detectar pontos de interesse
    sift = cv2.SIFT_create()

    # Detecta os pontos de interesse e calcula os descritores na imagem
    # keypoints: lista de pontos de interesse encontrados
    # descriptors: características (descritores) associadas a cada ponto de interesse
    keypoints, descriptors = sift.detectAndCompute(imagem, None)

    # Desenha os pontos de interesse na imagem redimensionada
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS: desenha os pontos com tamanho e orientação, indicando a força dos pontos de interesse
    imagem_sift = cv2.drawKeypoints(
        imagem_redimensionada, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return imagem_sift


sift_image = detectar_pontos_interesse_sift(imagem_cinza)
cv2.imshow('Detecção de Pontos de Interesse - SIFT', sift_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
