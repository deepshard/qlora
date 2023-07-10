from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer,
    StoppingCriteria,
    StoppingCriteriaList
)
import torch


from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)

model = AutoModelForCausalLM.from_pretrained(
        "huggyllama/llama-65b",
        cache_dir="/scratch/data",
        load_in_4bit=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        torch_dtype=(torch.bfloat16),
        trust_remote_code=True,
        use_auth_token=False
)

print("Adding peft..")
model = PeftModel.from_pretrained(model, "output/dreamshow-65b/checkpoint-1875/adapter_model")

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer,
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

tokenizer = AutoTokenizer.from_pretrained(
    "huggyllama/llama-65b",
)
if tokenizer._pad_token is None:
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token="[PAD]"),
        tokenizer=tokenizer,
        model=model,
    )
# LLaMA tokenizer may not have correct special tokens set.
# Check and add them if missing to prevent them from being parsed into different tokens.
# Note that these are present in the vocabulary.
# Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
print('Adding special tokens.')
tokenizer.add_special_tokens({
        "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
        "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
        "unk_token": tokenizer.convert_ids_to_tokens(
            model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
        ),
})

import pprint

def parse(a, b):
    return b.split(a)[1]

stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in ["User:"]]

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


def generate(t):
    inputs = tokenizer(t, return_tensors="pt").to("cuda:0")
    stop_words = ["User:"]
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    outputs = model.generate(**inputs, max_new_tokens=256, stopping_criteria=stopping_criteria)
    out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return out


#  eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id, pad_token_id=tokenizer.pad_token_id, force_words_ids=[[tokenizer.eos_token_id]]

text = '''
[TASK]:
The following is a role play conversation. Two characters engage in intimate sexual role play. Your task is to write one reply for the character named Lysandra. Remember to describe your actions in vivid detail.

[SCENARIO]:
Brion is a brown bald man. He is 26 years old and enjoys watching sluts get ravaged. He's currently at a sex club in Los Angeles and stumbles upon the user who is a slender model.

[TRAITS]:
Brion's personality is dark, playful, and dominating.

[CONVERSATION]:
Brion: *I see you across the room and I can't take my eyes off of you. You're so beautiful*
User: Hey, I'm Amith.
Brion:
'''




brion_first_out = generate(text, 256)
text = brion_first_out


