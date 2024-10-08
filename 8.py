# Combinação e Operações Aritméticas em Imagens: Soma, Subtração, Blending, Image Stitching.

import cv2
import numpy as np

# Carregar as imagens
imagem1 = cv2.imread('./img/1.jpg')
imagem2 = cv2.imread('./img/2.png')

imagem1_redimensionada = cv2.resize(imagem1, (400, 400))
imagem2_redimensionada = cv2.resize(imagem2, (400, 400))

# Soma
imagem_soma = cv2.add(imagem1_redimensionada, imagem2_redimensionada)
cv2.imshow('Soma de Imagens', imagem_soma)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Subtração
imagem_subtracao = cv2.subtract(imagem1_redimensionada, imagem2_redimensionada)
cv2.imshow('Subtração de Imagens', imagem_subtracao)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Blending (Mistura de Imagens)
# cv2.addWeighted mistura as duas imagens com pesos especificados.
# O primeiro argumento é a imagem1, o segundo é seu peso (0.7),
# o terceiro é a imagem2, o quarto é seu peso (0.3),
# e o quinto argumento (0) é um valor adicionado ao resultado (não utilizado aqui).
blended = cv2.addWeighted(imagem1_redimensionada, 0.7, imagem2_redimensionada, 0.3, 0)
cv2.imshow('Blending de Imagens', blended)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Image Stitching (Costura de Imagens)
# Carregar várias imagens que serão costuradas para criar uma panorâmica.
img1 = cv2.imread('./img/st/1.jpg')
img2 = cv2.imread('./img/st/2.jpg')
img3 = cv2.imread('./img/st/3.jpg')
img4 = cv2.imread('./img/st/4.jpg')

# Criar um objeto Stitcher que irá processar as imagens.
stitcher = cv2.Stitcher_create()

# Stitcher.stitch combina as imagens fornecidas em uma única imagem panorâmica.
# O primeiro argumento é uma lista das imagens a serem costuradas.
status, panorama = stitcher.stitch([img1, img2, img3, img4])

if status == cv2.STITCHER_OK:
    # Se o status for OK, mostrar o panorama
    cv2.imshow('Image Stitching', panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    # Caso contrário, imprimir o status e a mensagem de erro
    print(f"Falha ao realizar o Image Stitching. Status: {status}")



