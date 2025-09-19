# Arquivo de Configurações

# --- Parâmetros de Dados ---
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_CHANNELS = 3 # RGB

# Caminhos para os dados (ajuste conforme necessário)
DATA_PATH = "data/processed/"
RAW_DATA_PATH = "data/raw/"
MODEL_OUTPUT_PATH = "exported_models/"

# --- Parâmetros de Treinamento ---
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.15