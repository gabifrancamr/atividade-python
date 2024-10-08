# Leitura e Exibição de Imagens: Carregar imagens de diferentes formatos (JPEG, PNG, BMP) usando cv2.imread(). 
# Exibir as imagens carregadas utilizando cv2.imshow().

import cv2

imagem1 = cv2.imread('./img/1.jpg')
cv2.imshow("Imagem 1", imagem1)
cv2.waitKey(0)
cv2.destroyAllWindows()

imagem2 = cv2.imread('./img/2.png')
cv2.imshow("Imagem 2", imagem2)
cv2.waitKey(0)
cv2.destroyAllWindows()

imagem3 = cv2.imread('./img/3.webp')
cv2.imshow("Imagem 3", imagem3)
cv2.waitKey(0)
cv2.destroyAllWindows()
