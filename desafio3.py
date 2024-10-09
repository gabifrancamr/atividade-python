# cv2.cvtColor converte a imagem colorizada para o formato RGB.

import cv2

# Carregar uma imagem em preto e branco
gray_image = cv2.imread('./img/3.webp', cv2.IMREAD_GRAYSCALE)

# Aplica um mapa de cores à imagem em escala de cinza
# cv2.applyColorMap aplica uma paleta de cores (neste caso, JET) à imagem em escala de cinza
# O resultado é uma imagem colorida que representa os valores da imagem original em cores.
colorized_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)

# cv2.cvtColor converte a imagem colorizada para o formato RGB.
img_rgb = cv2.cvtColor(colorized_image, cv2.COLOR_BGR2RGB)

cv2.imshow("Colorização Simples", img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
