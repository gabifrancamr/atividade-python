import cv2

# Carregar uma imagem em preto e branco
gray_image = cv2.imread('./img/3.webp', cv2.IMREAD_GRAYSCALE)

# Colorização simples 
colorized_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)

img_rgb = cv2.cvtColor(colorized_image, cv2.COLOR_BGR2RGB)

cv2.imshow("Colorização Simples", img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
