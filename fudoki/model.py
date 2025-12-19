from transformers import AutoModelForCausalLM
from fudoki.janus.models import MultiModalityCausalLM


def instantiate_model(pretrained_weight_path):

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        pretrained_weight_path, trust_remote_code=True
    )
    model = vl_gpt
    return model
