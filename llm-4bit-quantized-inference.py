# -*- coding: utf-8 -*-

!pip install transformers torch accelerate bitsandbytes

!pip install -q transformers accelerate bitsandbytes huggingface_hub

from google.colab import userdata
from huggingface_hub import login

chave = userdata.get('HF_TOKEN')
login(token=chave)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

# 1. Configuração da Quantização (4-bits)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                  # Ativa a quantização para 4-bits
    bnb_4bit_use_double_quant=True,     # Quantização aninhada para salvar mais memória
    bnb_4bit_quant_type="nf4",          # Formato otimizado (NormalFloat4)
    bnb_4bit_compute_dtype=torch.bfloat16 # Mantém a computação rápida
)

print("Carregando o Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Carregando o Modelo quantizado (isso pode levar alguns minutos)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto" # Distribui automaticamente o modelo na GPU
)
print("Modelo carregado com sucesso!")

# Definindo a estrutura da conversa
prompt = "Explique, em um parágrafo, a vantagem de usar quantização em modelos de Inteligência Artificial."
messages = [
    {"role": "user", "content": prompt}
]

# 1. Aplica o template do Mistral (adicionando a tag que avisa o modelo para responder)
prompt_formatado = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 2. Transforma o texto formatado no dicionário de Tensores (input_ids e attention_mask)
inputs = tokenizer(prompt_formatado, return_tensors="pt").to("cuda")

# 3. Gerando a resposta (usamos **inputs para desempacotar o dicionário corretamente)
print("Gerando resposta...\n")
outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id # Boa prática para evitar warnings
)

# 4. Decodificando a resposta (pegamos apenas os tokens gerados depois do prompt)
tokens_gerados = outputs[0][inputs['input_ids'].shape[-1]:]
resposta_texto = tokenizer.decode(tokens_gerados, skip_special_tokens=True)

print(resposta_texto)
