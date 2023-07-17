import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaForCausalLM


def merge_weights(adapter_dir: str, output_dir: str, base_model: str):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        cache_dir="/scratch/data",
        torch_dtype=(torch.bfloat16),
        trust_remote_code=True,
        use_auth_token=False,
    ).to("cpu")

    first_weight = base_model.model.layers[0].self_attn.q_proj.weight
    first_weight_old = first_weight.clone()

    print("Adding peft...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("Added peft...")
    lora_weight = peft_model.base_model.model.model.layers[
        0
    ].self_attn.q_proj.weight
    assert torch.allclose(first_weight_old, first_weight)

    print("Merging weights...")
    lora_model = peft_model.merge_and_unload()
    lora_model.train(False)

    # did we do anything?
    assert not torch.allclose(first_weight_old, first_weight)

    print("Making state dict..")
    lora_model_sd = lora_model.state_dict()
    deloreanized_sd = {
        k.replace("base_model.model.", ""): v
        for k, v in lora_model_sd.items()
        if "lora" not in k
    }

    print("Saving to disk..")
    LlamaForCausalLM.save_pretrained(
        base_model, output_dir, state_dict=deloreanized_sd, max_shard_size="9GB"
    )
