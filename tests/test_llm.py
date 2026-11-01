import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from unittest.mock import patch
import torch
import llm_4bit_inference as m


def test_build_messages_without_system():
    assert m.build_messages("oi") == [{"role": "user", "content": "oi"}]


def test_build_messages_injects_system_into_user():
    msgs = m.build_messages("pergunta", "regras")
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "regras\n\npergunta"


def test_dtype_t4_uses_float16():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_capability", return_value=(7, 5)):
        assert m.pick_compute_dtype() == torch.float16


def test_dtype_ampere_uses_bfloat16():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_capability", return_value=(8, 0)):
        assert m.pick_compute_dtype() == torch.bfloat16
