import pytest
import torch
from ecg_project.models.lora import adapter_state,load_adapter


def test_qwen_patch_alignment_gradients_and_reload():
    pytest.importorskip('transformers')
    from transformers import Qwen3Config,Qwen3Model
    from ecg_project.models.qwen_delineator import QwenDelineator
    config=Qwen3Config(hidden_size=32,intermediate_size=64,num_hidden_layers=1,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,vocab_size=8)
    model=QwenDelineator(Qwen3Model(config),patch_size=7,rank=2,alpha=4)
    x=torch.randn(2,1,53)
    y=torch.randint(0,4,(2,53));y[:,:10]=-100
    out=model(x)
    assert out.shape==(2,4,53)
    torch.nn.functional.cross_entropy(out,y,ignore_index=-100).backward()
    assert model.input_adapter[0].weight.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if 'lora_B' in n)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    model.eval();expected=model(x).detach();state=adapter_state(model)
    with torch.no_grad():model.head.weight.zero_()
    load_adapter(model,state);torch.testing.assert_close(model(x),expected)


def test_real_nf4_adapter_backward():
    pytest.importorskip('bitsandbytes')
    if not torch.cuda.is_available():pytest.skip('CUDA NF4 smoke')
    from torch import nn
    from ecg_project.models.hubert import quantize_backbone
    from ecg_project.models.lora import inject_lora
    model=nn.ModuleDict({'q_proj':nn.Linear(64,64)})
    quantize_backbone(model);inject_lora(model,rank=2,alpha=4)
    model.cuda();x=torch.randn(4,64,device='cuda',requires_grad=True)
    model['q_proj'](x).square().mean().backward()
    assert model['q_proj'].base.weight.quant_state.quant_type=='nf4'
    assert model['q_proj'].lora_B.grad.abs().sum()>0
    assert model['q_proj'].base.weight.grad is None
