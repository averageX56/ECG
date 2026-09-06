import numpy as np
import pytest
import torch
from torch import nn
from types import SimpleNamespace
from ecg_project.models.lora import LoRALinear,adapter_state,load_adapter
from ecg_project.processing.hubert import prepare_signal,LEADS


def test_lora_zero_update_frozen_base_and_checkpoint():
    torch.manual_seed(42)
    base=nn.Linear(7,5);layer=LoRALinear(base,rank=2,alpha=4,dropout=0)
    x=torch.randn(3,7);original=base.weight.detach().clone()
    torch.testing.assert_close(layer(x),base(x))
    optimizer=torch.optim.SGD([p for p in layer.parameters() if p.requires_grad],lr=.1)
    layer(x).square().mean().backward()
    assert base.weight.grad is None
    assert layer.lora_B.grad.abs().sum()>0
    optimizer.step()
    torch.testing.assert_close(original,base.weight)
    saved=adapter_state(layer);expected=layer(x).detach()
    with torch.no_grad():layer.lora_B.zero_()
    load_adapter(layer,saved)
    torch.testing.assert_close(layer(x),expected)
    assert set(saved)=={'lora_A','lora_B'}


def test_hubert_preprocessing_lead_order_and_nonfinite():
    rng=np.random.default_rng(4);x=rng.normal(size=(5000,12))
    rec=SimpleNamespace(signal=x,fs=500,leads=LEADS)
    result=prepare_signal(rec)
    assert result.shape==(2,6000) and result.dtype==np.float32 and np.isfinite(result).all()
    perm=rng.permutation(12)
    swapped=SimpleNamespace(signal=x[:,perm],fs=500,leads=[LEADS[i] for i in perm])
    np.testing.assert_allclose(prepare_signal(swapped),result)
    rec.signal[0,0]=np.nan
    with pytest.raises(ValueError,match='Non-finite'):prepare_signal(rec)


def test_tiny_hubert_lora_backprop_and_reload():
    pytest.importorskip('transformers')
    from transformers import HubertConfig,HubertModel
    from ecg_project.models.hubert import HubertClassifier
    config=HubertConfig(hidden_size=24,num_hidden_layers=1,num_attention_heads=4,intermediate_size=48,
        conv_dim=(16,16),conv_kernel=(5,3),conv_stride=(2,2),num_conv_pos_embedding_groups=4,
        num_conv_pos_embeddings=8,apply_spec_augment=False)
    model=HubertClassifier(HubertModel(config),3,rank=2,alpha=4,checkpointing=True)
    model.train();x=torch.randn(2,2,200);loss=model(x).square().mean();loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if 'lora_B' in n)
    assert all(p.grad is None for n,p in model.named_parameters() if not p.requires_grad)
    model.eval();state=adapter_state(model);before=model(x)
    load_adapter(model,state);torch.testing.assert_close(model(x),before)
