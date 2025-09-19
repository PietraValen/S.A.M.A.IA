# Arquivo para a definição da arquitetura do modelo (U-Net)

import tensorflow as tf
from tensorflow.keras import layers
from src.config import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS

def unet_model(input_size=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)):
    """
    Define a arquitetura U-Net.
    Este é um modelo de exemplo e pode ser ajustado.
    """
    inputs = tf.keras.Input(input_size)
    
    # Exemplo simples de uma camada convolucional
    # A arquitetura completa da U-Net será implementada aqui
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    
    # A saída deve ter 1 canal (máscara binária) e ativação sigmóide
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c1)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    print("Modelo U-Net criado com sucesso.")
    model.summary()
    
    return model

if __name__ == '__main__':
    # Teste rápido para ver se o modelo é criado corretamente
    model = unet_model()