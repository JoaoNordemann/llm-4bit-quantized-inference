"""Inferência de LLM quantizado em 4-bit (NF4 + double quant) com BitsAndBytes."""
from __future__ import annotations

import argparse
import os

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"


def hf_login() -> None:
    """Lê HF_TOKEN do ambiente (fallback para Colab Secrets). Nunca hardcode o token."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:  # fallback opcional: só funciona dentro do Colab
            from google.colab import userdata  # type: ignore
            token = userdata.get("HF_TOKEN")
        except Exception:
            token = None
    if token:
        login(token=token)


def pick_compute_dtype() -> torch.dtype:
    """bfloat16 só em GPUs Ampere+ (compute capability >= 8). T4 (7.5) usa float16."""
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=pick_compute_dtype(),
    )


def load_model(model_id: str = DEFAULT_MODEL_ID):
    if not torch.cuda.is_available():
        raise RuntimeError("bitsandbytes 4-bit exige GPU CUDA.")
    hf_login()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=build_bnb_config(),
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def build_messages(user_prompt: str, system_prompt: str | None = None) -> list[dict]:
    """Mistral v0.2 não aceita role 'system': injeta as instruções na mensagem user."""
    content = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
    return [{"role": "user", "content": content}]


@torch.inference_mode()
def generate(
    tokenizer,
    model,
    user_prompt: str,
    system_prompt: str | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    prompt = tokenizer.apply_chat_template(
        build_messages(user_prompt, system_prompt),
        tokenize=False,
        add_generation_prompt=True,
    )
    # add_special_tokens=False: o chat template já inclui <s>
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)  # decodificação greedy, sem warnings

    outputs = model.generate(**inputs, **gen_kwargs)
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--prompt",
        default="Explique, em um parágrafo, a vantagem de usar quantização em modelos de IA.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_id)
    print(f"Modelo carregado | compute dtype: {pick_compute_dtype()} | "
          f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(generate(tokenizer, model, args.prompt,
                   max_new_tokens=args.max_new_tokens, temperature=args.temperature))


if __name__ == "__main__":
    main()
