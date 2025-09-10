#  Wiki Técnica: Projeto S.A.M.A.IA (O Cientista de Dados)

---

## 📖 Introdução

### Visão Geral do Sistema
O projeto S.A.M.A.IA (Sistema de Análise e Monitoramento de Ambientes por Inteligência Artificial) é uma iniciativa de pesquisa e desenvolvimento focada na criação de um modelo de Deep Learning para a detecção de desmatamento em biomas brasileiros, como a Amazônia e o Cerrado.

O núcleo do sistema é um "cérebro" de Inteligência Artificial, treinado para analisar imagens de satélite e identificar com alta precisão áreas que sofreram desmatamento. O resultado final deste repositório não é um aplicativo de usuário final, mas sim um **arquivo de modelo treinado (`.h5`, `.pth`, etc.)**, que representa o ativo de maior valor intelectual do projeto e que poderá ser consumido por outros sistemas.

A abordagem é puramente científica e experimental, envolvendo as seguintes etapas:
1.  **Coleta e Tratamento de Dados**: Aquisição de um grande volume de imagens geográficas.
2.  **Pesquisa e Desenvolvimento**: Treinamento e avaliação de diferentes arquiteturas de redes neurais (ex: U-Net, Mask R-CNN).
3.  **Avaliação de Performance**: Análise rigorosa dos resultados para selecionar o modelo mais eficiente.
4.  **Exportação do Modelo**: Empacotamento do modelo treinado para uso em sistemas de produção.

### Objetivos do Projeto
*   **Primário**: Desenvolver um modelo de segmentação de imagens de alta acurácia para identificar polígonos de desmatamento em imagens de satélite.
*   **Secundário**: Criar um pipeline de MLOps robusto e replicável, desde o processamento dos dados até o treinamento e a avaliação do modelo.
*   **Terciário**: Contribuir para a pesquisa na área de sensoriamento remoto e IA aplicada a causas ambientais.
*   **Estratégico**: Gerar um ativo de propriedade intelectual de alto valor para o repositório.

### Público-Alvo
*   **Desenvolvedores de IA/ML**: Engenheiros e cientistas de dados que irão treinar, avaliar e manter os modelos.
*   **Engenheiros de Dados**: Profissionais responsáveis pelo pipeline de ingestão e tratamento das imagens de satélite.
*   **DevOps/MLOps**: Especialistas que irão automatizar o ciclo de vida do modelo, incluindo o deploy e o monitoramento.
*   **Pesquisadores**: Cientistas da área de geociências e computação que podem utilizar os resultados e o modelo para estudos acadêmicos.
*   **Usuários Finais (Indiretos)**: Outros sistemas e plataformas que consumirão o modelo treinado para fornecer insights sobre desmatamento a analistas ambientais, órgãos governamentais e ONGs.

---

## ⚙️ Funcionalidades do Sistema

### Módulos Principais
O sistema é organizado em módulos que representam as etapas do pipeline de Machine Learning.

| Módulo | Arquivo(s) Principal(is) | Descrição da Funcionalidade |
| :--- | :--- | :--- |
| 📦 **Carregamento de Dados** | `src/dataLoader.py` | Responsável por carregar as imagens de satélite e suas respectivas máscaras de desmatamento. O módulo deve ser capaz de lidar com formatos geoespaciais (ex: GeoTIFF) e realizar o split inicial entre treino, validação e teste. |
| ✨ **Pré-processamento e Features** | `notebooks/explorerData.ipynb` | Embora não haja um script dedicado, esta funcionalidade envolve a normalização dos pixels, o aumento de dados (data augmentation) como rotações e flips, e a extração de features relevantes das imagens, como índices de vegetação (NDVI). |
| 🧠 **Arquitetura do Modelo** | `src/model.py` | Define a arquitetura da rede neural convolucional a ser utilizada. A escolha principal é uma arquitetura de segmentação semântica, como a U-Net, ideal para tarefas de pixel-a-pixel. |
| 🏋️ **Treinamento** | `src/train.py` | Orquestra o processo de treinamento. Este módulo alimenta o modelo com os dados de treino, calcula a função de perda (loss function), e ajusta os pesos da rede através de um otimizador. O progresso é salvo em checkpoints. |
| 📊 **Avaliação** | `src/evaluate.py` | Mede a performance do modelo treinado utilizando o conjunto de dados de validação/teste. As métricas principais incluem IoU (Intersection over Union), Dice Coefficient, Acurácia e F1-Score. |
| 🚀 **Exportação do Modelo** | `exportModels/modelSegmentacao.py` | Salva o modelo treinado e seus pesos em um formato padronizado (ex: HDF5 para Keras/TensorFlow) para que possa ser facilmente carregado e utilizado para inferência em outros ambientes. |
| 🔍 **Prototipagem e Análise** | `notebooks/*.ipynb` | Ambiente interativo para exploração de dados, prototipagem de modelos e análise de resultados. Essencial para o ciclo de pesquisa e desenvolvimento. |

---

## 🛠️ Configuração do Ambiente

### Instalação Passo a Passo
O ambiente de desenvolvimento pode ser configurado seguindo os passos abaixo. É recomendado o uso de um ambiente virtual Python (`venv`) para isolar as dependências.

```bash
# 1. Clone o repositório
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>

# 2. Crie e ative um ambiente virtual
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate

# 3. Instale as dependências
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Opcional) Configure o Jupyter Lab para o novo kernel
pip install ipykernel
python -m ipykernel install --user --name=samaia-kernel

echo "Ambiente configurado com sucesso!"
```

### Dependências e Bibliotecas
A lista completa de dependências está no arquivo `requirements.txt`. As bibliotecas mais críticas são:

| Biblioteca | Versão Mínima | Propósito no Projeto |
| :--- | :--- | :--- |
| `tensorflow` | `~2.x` | Framework principal para construção e treinamento da rede neural. |
| `scikit-learn` | `~1.x` | Utilizado para cálculo de métricas de avaliação e outras utilidades de ML. |
| `rasterio` | `~1.x` | Essencial para a leitura e escrita de dados de raster geoespacial (imagens de satélite). |
| `geopandas` | `~0.10.x` | Usado para manipular dados vetoriais (ex: polígonos de áreas desmatadas em formato Shapefile). |
| `Pillow` | `~9.x` | Biblioteca para manipulação básica de imagens. |
| `numpy` / `pandas` | `~1.2x` / `~1.4x` | Manipulação de dados numéricos e tabulares. |
| `matplotlib` | `~3.x` | Geração de gráficos e visualizações. |
| `jupyterlab` | `~3.x` | Ambiente de notebook para exploração e análise. |

### Configuração de Variáveis de Ambiente
Embora não haja um arquivo `.env` no projeto, é uma boa prática gerenciar caminhos de dados e hiperparâmetros através de variáveis de ambiente. Um arquivo `.env.example` seria assim:

```properties
# Exemplo de .env (NÃO EXISTE NO PROJETO, APENAS SUGESTÃO)
# =======================================================

# Caminhos para os dados
RAW_DATA_PATH=data/raw/
PROCESSED_DATA_PATH=data/processed/
MODEL_OUTPUT_PATH=models/

# Hiperparâmetros de Treinamento
EPOCHS=50
BATCH_SIZE=16
LEARNING_RATE=0.001
```
Atualmente, essas configurações são gerenciadas diretamente no código (ex: `src/config.py`), mas a centralização em um arquivo `.env` é uma melhoria recomendada.

---

## 🚀 Uso e Execução

### Modo Desenvolvimento
No modo de desenvolvimento, o foco está na experimentação e análise interativa. Os notebooks Jupyter são a principal ferramenta.

```bash
# Inicie o Jupyter Lab para explorar os dados ou prototipar modelos
jupyter lab notebooks/
```

### Modo Produção (Treinamento e Avaliação)
Para executar o pipeline de forma automatizada, use os scripts Python no diretório `src/`.

**1. Treinar o Modelo**
Execute o script `train.py`. Ele irá carregar os dados, construir o modelo e iniciar o treinamento.

```bash
# Exemplo de comando para iniciar o treinamento
python src/train.py --input_data "data/processed/" --model_output "models/" --epochs 100
```
> **Nota**: Os argumentos de linha de comando são hipotéticos e precisam ser implementados com `argparse` ou similar. Atualmente, os caminhos podem estar fixos no código.

**2. Avaliar o Modelo**
Após o treinamento, use `evaluate.py` para gerar as métricas de performance em um conjunto de teste.

```bash
# Exemplo de comando para avaliação
python src/evaluate.py --model_path "models/deforestation_model.h5" --test_data "data/test/"
```

**3. Exportar o Modelo para Inferência**
Execute o script de exportação para salvar o modelo final.

```bash
# Exemplo de comando para exportação
python exportModels/modelSegmentacao.py --trained_model "models/deforestation_model_final.h5" --export_path "release/final_model.h5"
```

---

## 📂 Estrutura do Projeto

A estrutura de diretórios foi projetada para separar claramente as diferentes preocupações do projeto.

```
.
├── DOC.md                  # Esta documentação
├── LICENSE                 # Licença do projeto
├── README.md               # README geral
├── exportModels/           # Scripts para exportar o modelo treinado
│   └── modelSegmentacao.py # Lógica para salvar o modelo para produção
├── notebooks/              # Notebooks para análise e prototipagem
│   ├── avacuationResult.ipynb # (Typo) Análise de resultados da avaliação
│   ├── explorerData.ipynb     # Exploração inicial dos dados (EDA)
│   └── prototypeModel.ipynb   # Prototipagem da arquitetura do modelo
├── requirements.txt        # Lista de dependências Python
└── src/                    # Código-fonte principal do projeto
    ├── config.py           # Arquivo de configuração (parâmetros, caminhos)
    ├── dataLoader.py       # Scripts para carregamento de dados
    ├── evaluate.py         # Script para avaliar o modelo treinado
    ├── model.py            # Definição da arquitetura da rede neural
    └── train.py            # Script principal para o treinamento do modelo
```

### Fluxo de Dados Interno
1.  **Origem**: As imagens brutas (`.tif`, `.png`) e máscaras de desmatamento residem em um diretório de dados (ex: `data/raw/`).
2.  **Carregamento**: `dataLoader.py` lê as imagens e máscaras.
3.  **Processamento**: As imagens são normalizadas e possivelmente aumentadas (data augmentation).
4.  **Treinamento**: `train.py` utiliza os dados processados para treinar o modelo definido em `model.py`. Checkpoints do modelo são salvos em `models/`.
5.  **Avaliação**: `evaluate.py` carrega um modelo salvo e o conjunto de teste para calcular as métricas de performance. Os resultados podem ser salvos em `reports/`.
6.  **Exportação**: `modelSegmentacao.py` pega o melhor modelo treinado e o salva em um formato final para deploy.

---

## 📡 APIs e Endpoints

Este projeto não expõe uma API REST/HTTP tradicional. A "API" do sistema é, na verdade, a interface programática do modelo treinado.

### Interface do Modelo (Contrato de Inferência)
Para usar o modelo, um sistema externo deve:
1.  Carregar o arquivo do modelo (ex: `.h5`).
2.  Pré-processar a imagem de entrada para que ela tenha o mesmo formato (dimensões, normalização de pixels) que os dados de treinamento.
3.  Chamar o método de predição do modelo.

**Exemplo de Execução Programática (Python)**

```python
# [CÓDIGO EXEMPLO AQUI]
# O código abaixo é uma representação de como o modelo seria consumido.

import tensorflow as tf
import rasterio
import numpy as np

# 1. Carregar o modelo treinado
MODEL_PATH = "release/final_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Função para pré-processar a imagem de entrada
def preprocess_image(image_path):
    with rasterio.open(image_path) as src:
        # Supondo que o modelo espera 3 canais (RGB) e tamanho 256x256
        image = src.read([1, 2, 3])
        # Transpor para (altura, largura, canais)
        image = np.transpose(image, (1, 2, 0))
        # Normalizar pixels para o intervalo [0, 1]
        image = image / 255.0
        # Adicionar dimensão do batch
        return np.expand_dims(image, axis=0)

# 2. Carregar e pré-processar uma nova imagem de satélite
new_image_path = "path/to/new_satellite_image.tif"
processed_image = preprocess_image(new_image_path)

# 3. Realizar a predição
predicted_mask = model.predict(processed_image)

# 4. Pós-processar a máscara de saída
# (ex: converter probabilidades para uma máscara binária com um limiar de 0.5)
binary_mask = (predicted_mask[0, :, :, 0] > 0.5).astype(np.uint8)

print("Máscara de desmatamento predita com sucesso.")
print(f"Shape da máscara: {binary_mask.shape}")
```

### Respostas de Sucesso/Erro
*   **Sucesso**: O método `predict()` retorna um array NumPy contendo a máscara de segmentação predita. O shape da saída será algo como `(1, altura, largura, 1)`.
*   **Erro**: Erros ocorrerão se a imagem de entrada não tiver as dimensões ou o número de canais esperados pelo modelo, resultando em uma `ValueError` do TensorFlow/Keras.

---

## 🔒 Segurança

### Camadas de Proteção
A segurança neste projeto está focada na integridade dos dados, do código e do modelo treinado.

| Camada | Estratégia de Proteção |
| :--- | :--- |
| **Código-Fonte** | O acesso ao repositório Git deve ser restrito a pessoal autorizado. Uso de branches protegidas (ex: `main`) para evitar commits diretos. |
| **Dados** | Os datasets de imagens de satélite e máscaras devem ser armazenados em um local seguro (ex: S3 Bucket privado) com políticas de acesso restritas. |
| **Modelo Treinado** | O modelo (`.h5`) é o principal ativo intelectual. Deve ser tratado como um segredo, com controle de versão e acesso restrito. Não deve ser commitado diretamente no Git se for muito grande. |
| **Dependências** | Uso de ferramentas como `pip-audit` ou Snyk para escanear `requirements.txt` em busca de vulnerabilidades conhecidas. |

### Checklist de Segurança
- [ ] O acesso ao repositório está restrito?
- [ ] A branch `main` está protegida contra pushes diretos?
- [ ] Os dados sensíveis (se houver) não estão commitados no repositório?
- [ ] As dependências foram escaneadas em busca de vulnerabilidades?
- [ ] O modelo treinado está armazenado em um artefato seguro (ex: AWS S3, Google Cloud Storage)?

---

## ⚡ Performance

### Benchmark do Sistema
A performance é avaliada em duas frentes: velocidade de treinamento/inferência e acurácia do modelo.

**Métricas de Acurácia**
| Métrica | Descrição | Objetivo |
| :--- | :--- | :--- |
| **Intersection over Union (IoU)** | Mede a sobreposição entre a máscara predita e a real. A métrica mais importante para segmentação. | Maximizar (alvo > 0.80) |
| **Dice Coefficient** | Similar ao IoU, muito usada em segmentação de imagens médicas e de satélite. | Maximizar |
| **Acurácia (Pixel-wise)** | Percentual de pixels classificados corretamente. Pode ser enganosa em datasets desbalanceados. | Maximizar |
| **Precisão e Recall** | Medem a taxa de falsos positivos e falsos negativos, respectivamente. | Balancear |

**Métricas de Velocidade**
| Métrica | Descrição | Objetivo |
| :--- | :--- | :--- |
| **Tempo de Treinamento** | Horas/dias necessários para treinar o modelo em um determinado hardware (ex: GPU V100). | Minimizar |
| **Latência de Inferência** | Tempo (em milissegundos) para processar uma única imagem. | Minimizar (crítico para uso em tempo real) |

### Estratégias de Otimização
*   **Hardware**: Utilizar GPUs (NVIDIA V100, A100) para acelerar o treinamento.
*   **Software**: Usar `tf.data` para criar pipelines de entrada de dados eficientes e assíncronos.
*   **Modelo**: Aplicar técnicas como quantização (reduzir a precisão dos pesos do modelo) ou pruning (remover neurônios redundantes) para diminuir o tamanho do modelo e acelerar a inferência, com uma pequena perda de acurácia.

---

## 📊 Monitoramento e Logs

### Endpoints de Saúde
Por não ser um serviço web, o projeto não possui um endpoint `/health`. O "estado de saúde" do sistema é verificado pela capacidade de executar os scripts com sucesso. Um script de "smoke test" pode ser criado para verificar se o ambiente e os dados estão corretos.

### Estratégia de Coleta de Logs
O logging deve ser implementado usando o módulo `logging` do Python.

*   **Durante o Treinamento**: Registrar a perda (loss) e as métricas de acurácia a cada época. Salvar os logs em um arquivo (ex: `training.log`). Ferramentas como TensorBoard ou Weights & Biases são altamente recomendadas para visualização em tempo real.
*   **Durante a Inferência**: Registrar o ID da imagem processada, o tempo de processamento e o shape da máscara de saída.
*   **Erros**: Capturar exceções e registrar o stack trace completo para facilitar o debug.

**Exemplo de Log Estruturado (JSON)**
```json
{
  "timestamp": "2023-10-27T10:00:00Z",
  "level": "INFO",
  "module": "train.py",
  "event": "epoch_end",
  "details": {
    "epoch": 10,
    "loss": 0.1523,
    "iou_score": 0.785
  }
}
```

---

## 🧪 Testes

### Tipos de Testes
*   **Testes Unitários**: Foco em pequenas unidades de código, como funções individuais (ex: uma função de pré-processamento). Devem ser rápidos e não depender de dados externos.
*   **Testes de Integração**: Verificam a interação entre os módulos, por exemplo, se o `dataLoader` produz dados no formato que o `model` espera.
*   **Testes de Regressão (Validação)**: Um conjunto fixo de dados de validação é usado para avaliar o modelo após cada treinamento. Se a métrica principal (ex: IoU) piorar significativamente, o novo modelo é rejeitado. Isso previne a regressão de performance.

### Relatórios de Cobertura
A cobertura de testes (`code coverage`) pode ser medida com a biblioteca `pytest-cov`. O objetivo é garantir que as partes críticas do código (carregamento de dados, lógica do modelo) sejam testadas.

```bash
# Exemplo de execução de testes com coverage
pytest --cov=src
```

---

## 🐞 Troubleshooting

### Lista de Erros Frequentes
| Erro | Causa Provável | Solução |
| :--- | :--- | :--- |
| `ResourceExhaustedError` (OOM) | O batch size é muito grande e a GPU fica sem memória (Out Of Memory). | Reduzir o `BATCH_SIZE` no script de treinamento. Otimizar o pipeline de dados. |
| `ValueError: Mismatch in shapes` | As dimensões da imagem de entrada ou da máscara não correspondem ao que o modelo espera. | Verificar a etapa de pré-processamento. Garantir que todas as imagens sejam redimensionadas para o mesmo tamanho. |
| `ModuleNotFoundError` | Uma dependência não está instalada no ambiente Python. | Execute `pip install -r requirements.txt` no ambiente virtual correto. |
| `Loss is NaN` (Not a Number) | A taxa de aprendizado (`learning rate`) é muito alta, causando instabilidade numérica. | Reduzir o `LEARNING_RATE` no otimizador. Verificar se há valores nulos ou infinitos nos dados de entrada. |

---

## 💾 Banco de Dados e Backup

### Modelo de Dados
O "banco de dados" deste projeto é o sistema de arquivos. O modelo de dados consiste em:

*   **Imagens de Satélite (Rasters)**: Arquivos `.tif` ou `.png` com múltiplos canais (ex: RGB, Infravermelho).
*   **Máscaras de Desmatamento (Rasters)**: Imagens em escala de cinza ou binárias onde cada pixel representa uma classe (ex: 0 para "não desmatado", 1 para "desmatado").
*   **Modelo Treinado (Artefato)**: Arquivo `.h5` ou similar contendo a arquitetura e os pesos do modelo.

### Estratégia de Backup/Restauração
*   **Código**: O Git é o sistema de backup. Pushes regulares para um repositório remoto (GitHub, GitLab) são essenciais.
*   **Dados**: Os datasets devem ser versionados (ex: DVC - Data Version Control) e armazenados em um storage de objetos resiliente (AWS S3, Google Cloud Storage) com versionamento habilitado.
*   **Modelos**: Os modelos treinados devem ser tratados como artefatos de build e armazenados em um registro de modelos (MLflow, Vertex AI Model Registry) ou em um bucket versionado.

---

## 🌐 Infraestrutura e Deploy

### CI/CD
Um pipeline de Integração Contínua e Deploy Contínuo (CI/CD) para este projeto de ML (MLOps) poderia ter as seguintes etapas:

1.  **On `git push`**:
    *   Executar testes unitários e de integração.
    *   Executar linters (ex: `flake8`, `black`) para garantir a qualidade do código.
2.  **On `merge to main` (ou tag)**:
    *   Disparar o script `src/train.py` para treinar o modelo com o dataset completo.
    *   Executar o script `src/evaluate.py` para avaliar o novo modelo.
    *   Se o novo modelo superar o antigo, promovê-lo e salvá-lo no registro de modelos.

**Exemplo de Pipeline (GitHub Actions - `.github/workflows/ci-cd.yml`)**
```yaml
# [EXEMPLO DE CÓDIGO YAML AQUI]
name: CI/CD Pipeline for S.A.M.A.IA

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest

  train:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: [self-hosted, gpu] # Requer um runner com GPU
    steps:
    - # ... steps para treinar o modelo ...
      name: Train model
      run: python src/train.py
    - # ... steps para salvar o modelo em um bucket ...
```

### Servidores/Ambientes Suportados
*   **Desenvolvimento**: Máquina local com Python e `venv`.
*   **Treinamento**: Instância de Cloud com GPU (ex: AWS EC2 P3, Google Cloud AI Platform).
*   **Deploy (Inferência)**: O modelo pode ser implantado em qualquer ambiente que suporte TensorFlow, desde um servidor na nuvem até dispositivos de edge.

---

## 🔗 Integrações

### Sistemas Externos Conectados
*   **Fonte de Dados**: O sistema se integra a fontes de imagens de satélite, que podem ser APIs de provedores (Planet, Maxar) ou datasets públicos (Landsat, Sentinel).
*   **Consumidores do Modelo**: O modelo treinado é projetado para ser integrado a outros sistemas, como plataformas de GIS (Geographic Information System), dashboards de monitoramento ambiental ou sistemas de alerta.

### Autenticação
A autenticação não se aplica diretamente ao modelo, mas sim à infraestrutura ao redor dele.
*   **Acesso aos Dados**: Se os dados estiverem em um bucket na nuvem, os scripts precisarão de credenciais (ex: chaves de acesso IAM da AWS) para acessá-los.
*   **APIs de Terceiros**: Se o sistema consumir dados de APIs externas, ele precisará gerenciar tokens de API (ex: API Key) de forma segura.

---

## 📌 Tabela Técnica Resumida

| Categoria | Detalhe |
| :--- | :--- |
| **Plataforma** | Pesquisa e Desenvolvimento de Modelo de IA |
| **Link Repositório** | `[INSERIR LINK DO GITHUB AQUI]` |
| **Tipo** | Modelo de Deep Learning (Segmentação de Imagens) |
| **Status** |  attivo |
| **Tecnologia Principal** | Python, TensorFlow, Rasterio, Geopandas |
| **Versão do Projeto** | v0.1.0 (Fase de Prototipagem) |
| **Build / Deploy** | Scripts Python, potencialmente automatizado com GitHub Actions |
| **Servidor / Ambiente** | Local (Dev), Cloud com GPU (Treinamento) |
| **Banco de Dados** | Sistema de Arquivos (Imagens, Máscaras) |
| **Host / IP** | N/A |
| **SSL / Segurança** | Gerenciamento de acesso ao repo e aos dados |
| **Backup** | Git para código; Storage de Objetos (S3, GCS) para dados/modelos |
| **Monitoramento** | Logs locais, TensorBoard (sugerido) |
| **Nível de Criticidade** | Alto (o modelo é o principal ativo intelectual) |
| **Integrações** | Fontes de dados de satélite (entrada), sistemas de GIS (saída) |
| **Wiki Técnica** | 📖 Incluído (este documento) |
| **Observações Gerais** | Projeto em fase inicial. O código-fonte principal ainda está em desenvolvimento. |

---

## 📚 Conclusão e Próximos Passos

### Roadmap do Projeto
*   **Q4 2023**: Finalizar a implementação dos scripts `src/` e realizar o primeiro ciclo de treinamento completo.
*   **Q1 2024**: Implementar um pipeline de CI/CD básico com GitHub Actions para automatizar testes e treinamento.
*   **Q2 2024**: Pesquisar e experimentar arquiteturas de modelo mais avançadas (ex: Attention U-Net).
*   **Q3 2024**: Desenvolver uma API simples (ex: com FastAPI) para servir o modelo e facilitar a integração.

### Melhorias Futuras
*   **Versionamento de Dados**: Implementar DVC para versionar os datasets.
*   **Experiment Tracking**: Integrar com MLflow ou Weights & Biases para um rastreamento mais robusto dos experimentos.
*   **Otimização**: Aplicar quantização e pruning para criar uma versão leve do modelo para deploy em edge.

### Sugestões de Escalabilidade
*   **Treinamento Distribuído**: Para datasets muito grandes, usar `tf.distribute.Strategy` para treinar o modelo em múltiplos nós/GPUs.
*   **Pipeline de Dados**: Migrar o processamento de dados para uma plataforma de processamento distribuído como o Apache Spark se o volume de dados se tornar um gargalo.
