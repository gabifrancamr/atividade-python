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
    imagem_float = np.float32(imagem)
    cantos_harris = cv2.cornerHarris(imagem_float, 2, 3, 0.04)
    imagem_harris = imagem_redimensionada.copy()
    imagem_harris[cantos_harris > 0.01 * cantos_harris.max()] = [0, 0, 255]  # Marcar os cantos em vermelho
    return imagem_harris

harris_corners = detectar_cantos_harris(imagem_cinza)
cv2.imshow('Detecção de Cantos - Harris', harris_corners)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. Detecção de Cantos - Shi-Tomasi (Good Features to Track)
def detectar_cantos_shi_tomasi(imagem):
    cantos_shi_tomasi = cv2.goodFeaturesToTrack(imagem, 100, 0.01, 10)
    cantos_shi_tomasi = np.int0(cantos_shi_tomasi)
    imagem_shi_tomasi = imagem_redimensionada.copy()
    for canto in cantos_shi_tomasi:
        x, y = canto.ravel()
        cv2.circle(imagem_shi_tomasi, (x, y), 5, (0, 255, 0), -1)  # Marcar os cantos em verde
    return imagem_shi_tomasi

shi_tomasi_corners = detectar_cantos_shi_tomasi(imagem_cinza)
cv2.imshow('Detecção de Cantos - Shi-Tomasi', shi_tomasi_corners)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. Detecção de Contornos
def detectar_contornos(imagem):
    bordas = cv2.Canny(imagem, 100, 200)
    contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imagem_contornos = imagem_redimensionada.copy()
    cv2.drawContours(imagem_contornos, contornos, -1, (255, 0, 0), 2)  # Desenhar contornos em azul
    return imagem_contornos

contours_image = detectar_contornos(imagem_cinza)
cv2.imshow('Detecção de Contornos', contours_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4. Detecção de Pontos de Interesse - SIFT
def detectar_pontos_interesse_sift(imagem):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(imagem, None)
    imagem_sift = cv2.drawKeypoints(imagem_redimensionada, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return imagem_sift

sift_image = detectar_pontos_interesse_sift(imagem_cinza)
cv2.imshow('Detecção de Pontos de Interesse - SIFT', sift_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
