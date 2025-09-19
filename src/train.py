# Script principal para o treinamento do modelo

import tensorflow as tf
from src.config import BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_OUTPUT_PATH
from src.data_loader import load_data
from src.model import unet_model

def main():
    """Função principal que orquestra o processo de treinamento."""
    print("Iniciando o processo de treinamento...")
    
    # 1. Carregar os dados
    images, masks = load_data()
    
    # 2. Criar o modelo
    model = unet_model()
    
    # 3. Compilar o modelo
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # 4. Treinar o modelo (aqui entrariam os dados reais)
    print("Iniciando o treinamento (com dados de exemplo)...")
    # model.fit(images, masks, batch_size=BATCH_SIZE, epochs=EPOCHS)
    print("Treinamento (simulado) concluído.")
    
    # 5. Salvar o modelo treinado
    model_path = f"{MODEL_OUTPUT_PATH}modelo_segmentacao_v1.h5"
    # model.save(model_path)
    print(f"Modelo salvo em: {model_path} (simulado)")

if __name__ == '__main__':
    main()