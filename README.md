# Inferência Otimizada de LLM com Quantização em 4-bits

Este repositório contém a implementação de um pipeline de inferência para modelos de linguagem de grande escala (LLMs) utilizando a biblioteca `transformers` da Hugging Face e técnicas de otimização de memória com `bitsandbytes`.

## Objetivo do Projeto
Demonstrar a viabilidade de carregamento e execução de modelos de 7 Bilhões de parâmetros (como o Mistral-7B) em hardwares com restrição de VRAM (como GPUs de 15GB), garantindo eficiência de infraestrutura sem perda significativa de qualidade de geração.

## Decisão Arquitetural: Por que 4-bit Quantization?
Carregar um modelo 7B em precisão padrão (`float16`) consome aproximadamente 14-15GB de VRAM, deixando pouca margem para o processamento de contexto (janela de tokens), o que causa erros de Out-Of-Memory (OOM). 

Ao utilizar o `BitsAndBytesConfig` com formato `nf4` (NormalFloat4) e dupla quantização, comprimimos os pesos do modelo diretamente no carregamento. O impacto:
* **Redução de VRAM:** De ~14GB para aproximadamente ~5GB.
* **Escalabilidade:** Permite rodar LLMs poderosos em instâncias de nuvem mais baratas (como a T4 do Google Colab ou AWS `g4dn`), reduzindo os custos operacionais da infraestrutura de IA.

## Como Executar
1. Instale as dependências:
   ```bash
   pip install transformers torch accelerate bitsandbytes huggingface_hub
