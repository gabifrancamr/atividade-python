# Manipulação de Arquivos: Salvar imagens processadas em diferentes formatos com cv2.imwrite().
import cv2

imagem = cv2.imread('./img/1.jpg')

imagem_processada = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
cv2.imwrite('imagem_processada.png', imagem_processada)

