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

model = cv2.dnn.readNetFromCaffe('./data/models/deploy.prototxt', './data/models/res10_300x300_ssd_iter_140000.caffemodel')

# Cria um blob a partir da imagem
img_copy = img_rgb.copy()

blob = cv2.dnn.blobFromImage(cv2.resize(
    img_copy, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# Define o blob como entrada para o modelo
model.setInput(blob)

# Executa uma passagem para frente (forward pass) através do modelo
detections = model.forward()

# Obtém as dimensões da imagem
(h, w) = img_copy.shape[:2]

# Itera sobre as detecções
for i in range(detections.shape[2]):
    # Obtém a confiança da detecção
    confidence = detections[0, 0, i, 2]

    # Define um limiar de confiança para considerar uma detecção válida
    confidence_threshold = 0.5

    # Verifica se a confiança é maior que o limiar
    if confidence > confidence_threshold:
        # Obtém as coordenadas do retângulo delimitador
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

        # Converte as coordenadas para inteiros
        (startX, startY, endX, endY) = box.astype("int")

        # Desenha o retângulo delimitador na imagem
        cv2.rectangle(img_copy, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Adiciona a confiança da detecção na imagem
        text = "Carlos"# "{:.2f}%".format(confidence * 100)
        cv2.putText(img_copy, text, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

# Neste ponto, 'img_copy' contém as detecções processadas
# sig.plot_image(img_copy)
plt.imshow(img_copy)
# plt.axis('off')  # Desligar os eixos
plt.show()

