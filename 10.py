# Manipulação de Arquivos: Salvar imagens processadas em diferentes formatos com cv2.imwrite().
import cv2

imagem = cv2.imread('./img/1.jpg')

# Processar a imagem: converter a imagem colorida (BGR) para escala de cinza.
# cv2.cvtColor converte a imagem para escala de cinza, armazenando o resultado em 'imagem_processada'.
imagem_processada = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Salvar a imagem processada em um arquivo no formato PNG.
# cv2.imwrite salva a imagem especificada pelo nome e formato fornecido. Neste caso, 'imagem_processada.png'.
cv2.imwrite('imagem_processada.png', imagem_processada)

