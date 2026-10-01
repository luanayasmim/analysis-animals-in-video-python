import cv2
from ultralytics import YOLO

# 1. Carrega o modelo pré-treinado do YOLOv8 (versão nano 'n', rápida e leve)
model = YOLO('yolov8n.pt')

# 2. Insira o caminho da sua imagem aqui
image_path = './image copy.png'

# 3. Realiza a detecção de objetos na imagem
results = model(image_path)

# Lista de classes de animais presentes no dataset COCO (padrão do YOLO)
# Você pode adicionar ou remover classes conforme a sua necessidade
animais_alvo = [
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
]

total_animais = 0
contagem_por_tipo = {}

# 4. Analisa os resultados e filtra apenas os animais
for r in results:
  for box in r.boxes:
    class_id = int(box.cls[0])
    class_name = model.names[class_id]

    if class_name in animais_alvo:
      total_animais += 1
      contagem_por_tipo[class_name] = (
          contagem_por_tipo.get(class_name, 0) + 1
      )

# 5. Exibe os resultados no console
print(f'Quantidade total de animais encontrados: {total_animais}')
if total_animais > 0:
  print('Detalhes por espécie:')
  for especie, qtd in contagem_por_tipo.items():
    print(f'- {especie}: {qtd}')

# 6. Opcional: Salva e exibe a imagem com as caixas delimitadoras desenhadas
imagem_com_deteccoes = results[0].plot()
cv2.imwrite('resultado_animais.jpg', imagem_com_deteccoes)
print("Imagem salva com sucesso como 'resultado_animais.jpg'!")