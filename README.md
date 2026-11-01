# LLM 4-bit Quantized Inference

Inferência de LLMs em 4-bit (NF4 + double quantization) com `bitsandbytes` e `transformers`, rodando em GPUs de consumo (ex.: T4 do Colab). Modelo padrão: `mistralai/Mistral-7B-Instruct-v0.2`.

## Decisões técnicas
| Tema | Decisão | Motivo |
|---|---|---|
| Quantização | NF4 + double quant | Menor uso de VRAM com perda mínima de qualidade |
| Compute dtype | `bfloat16` se compute capability >= 8, senão `float16` | T4 (7.5) não tem bf16 nativo |
| Amostragem | `do_sample=True` apenas se `temperature > 0` | `temperature` é ignorado sem sampling |
| System prompt | Injetado na mensagem `user` | Template do Mistral v0.2 não aceita role `system` |
| Segredos | `HF_TOKEN` via variável de ambiente | Nada de token no código |

## Uso
```bash
pip install -r requirements.txt
export HF_TOKEN=hf_xxx
python llm_4bit_inference.py --prompt "Explique quantização em 1 parágrafo." --max-new-tokens 150
```
Como biblioteca:
```python
from llm_4bit_inference import load_model, generate
tok, model = load_model()
print(generate(tok, model, "Pergunta", system_prompt="Responda em pt-BR.", temperature=0.2))
```

## Testes
```bash
pytest -q
```

## Roadmap
- [ ] Benchmark VRAM / latência / tokens/s: FP16 vs 8-bit vs NF4
- [ ] Perplexity (WikiText) por nível de quantização
- [ ] Fine-tuning QLoRA (peft + trl) e adapter publicado no Hugging Face

## Requisitos
GPU NVIDIA com CUDA (bitsandbytes 4-bit não roda em CPU).
