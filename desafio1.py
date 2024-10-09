# Reconhecimento de Objetos: Adicionar reconhecimento de objetos com um modelo pré-treinado.

import cv2
import numpy as np  
import matplotlib.pyplot as plt

# Carregando a imagem
img = cv2.imread('./img/imgg.png')

# Convertendo a imagem de BGR para RGB (OpenCV usa BGR por padrão)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Exibindo a imagem
# plt.imshow(img_rgb)
# plt.axis('off')  # Desligar os eixos
# plt.show()


# Carregando o modelo Caffe para detecção de rostos
# 'deploy.prototxt' contém a arquitetura do modelo e 'res10_300x300_ssd_iter_140000.caffemodel' contém os pesos treinados.
model = cv2.dnn.readNetFromCaffe('./data/models/deploy.prototxt', './data/models/res10_300x300_ssd_iter_140000.caffemodel')

# Faz uma cópia da imagem para evitar modificações na imagem original
img_copy = img_rgb.copy()

# Cria um blob a partir da imagem redimensionada
# O blob é uma representação da imagem que o modelo pode processar.
# cv2.resize(img_copy, (300, 300)) redimensiona a imagem para 300x300 pixels.
# O primeiro parâmetro é a imagem, o segundo é o novo tamanho (300, 300).
# 1.0 é a escala (não altera os valores dos pixels).
# (300, 300) é a dimensão da imagem de saída do blob.
# (104.0, 177.0, 123.0) é a média subtraída dos canais BGR da imagem (valores para normalização).
# Normalizar a imagem ajuda a melhorar a performance do modelo.
blob = cv2.dnn.blobFromImage(cv2.resize(
    img_copy, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# Define o blob como entrada para o modelo
model.setInput(blob)

# Executa uma passagem para frente (forward pass) através do modelo
detections = model.forward()

# Obtém as dimensões da imagem original (altura e largura)
# img_copy.shape[:2] retorna as dimensões da imagem: (altura, largura)
(h, w) = img_copy.shape[:2]

# Itera sobre as detecções retornadas pelo modelo
for i in range(detections.shape[2]):
    # Obtém a confiança da detecção para a i-ésima detecção
    confidence = detections[0, 0, i, 2]

    # Define um limiar de confiança para considerar uma detecção válida
    confidence_threshold = 0.5

    # Verifica se a confiança é maior que o limiar definido
    if confidence > confidence_threshold:
        # Obtém as coordenadas do retângulo delimitador para a detecção
        # As coordenadas são normalizadas e multiplicadas pelas dimensões da imagem original
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

        # Converte as coordenadas para inteiros para desenhar o retângulo
        (startX, startY, endX, endY) = box.astype("int")

        # Desenha o retângulo delimitador na imagem copiando
        cv2.rectangle(img_copy, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Adiciona a confiança da detecção na imagem (pode-se incluir o nome ou outra informação)
        text = "Carlos"  # "{:.2f}%".format(confidence * 100)  # Exemplo de como formatar a confiança
        cv2.putText(img_copy, text, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)


plt.imshow(img_copy)
# plt.axis('off')  # Desligar os eixos
plt.show()

