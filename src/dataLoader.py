# Arquivo para carregar e pré-processar os dados

import os
import numpy as np
from src.config import DATA_PATH

def load_data():
    """
    Função para carregar os dados de treinamento e validação.
    Esta função irá ler as imagens e máscaras da pasta /data/processed.
    """
    print(f"Carregando dados do caminho: {DATA_PATH}")
    
    # Lógica para carregar imagens e máscaras virá aqui
    # Por enquanto, retorna dados de exemplo
    
    # Exemplo: 10 imagens de 256x256 com 3 canais (RGB)
    dummy_images = np.random.rand(10, 256, 256, 3) 
    # Exemplo: 10 máscaras de 256x256 com 1 canal
    dummy_masks = np.random.randint(0, 2, (10, 256, 256, 1))
    
    print("Dados de exemplo gerados.")
    return dummy_images, dummy_masks

if __name__ == '__main__':
    images, masks = load_data()
    print(f"Formato das imagens: {images.shape}")
    print(f"Formato das máscaras: {masks.shape}")
    