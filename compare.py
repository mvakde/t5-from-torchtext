# This file checks if the T5 implementation matches other Huggingface's implementation

from transformers import T5Model, T5Tokenizer
import torch
from t5_architecture import T5_BASE 

# Input and output sequences
input_sentence = ["translate to Spanish: My name is Joe"]
output_sentence = ["Me llamo Joe"]

# Load your local T5 model and transform
t5_base = T5_BASE
transform = t5_base.transform()
tt_t5_model = t5_base.get_model()

# Load Hugging Face T5 model and tokenizer
hf_t5_model = T5Model.from_pretrained("t5-base")
hf_tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Tokenize inputs with your local transform
tokenized_sentence = transform(input_sentence)
tokenized_output = transform(output_sentence)

# Tokenize inputs with Hugging Face tokenizer
hf_tokenized_sentence = hf_tokenizer(input_sentence, return_tensors="pt").input_ids
hf_tokenized_output = hf_tokenizer(output_sentence, return_tensors="pt").input_ids

# Move the data to the same device as your model, if needed
device = next(tt_t5_model.parameters()).device
tokenized_sentence = tokenized_sentence.to(device)
tokenized_output = tokenized_output.to(device)
hf_tokenized_sentence = hf_tokenized_sentence.to(device)
hf_tokenized_output = hf_tokenized_output.to(device)


# Get the outputs from your local model
tt_output = tt_t5_model(encoder_tokens=tokenized_sentence, decoder_tokens=tokenized_output)

# Get the outputs from Hugging Face's model
hf_output = hf_t5_model(input_ids=hf_tokenized_sentence, decoder_input_ids=hf_tokenized_output, return_dict=True)

# Assertion: Compare the encoder outputs.
print(torch.all(tt_output["encoder_output"] == hf_output["encoder_last_hidden_state"]))
assert torch.allclose(tt_output["encoder_output"], hf_output["encoder_last_hidden_state"], atol=1e-5), "Encoder outputs do not match"

# Assertion: Compare the decoder outputs
print(torch.all(tt_output["decoder_output"] == hf_output["last_hidden_state"]))
assert torch.allclose(tt_output["decoder_output"], hf_output["last_hidden_state"], atol=1e-5), "Decoder outputs do not match"

print("Outputs match within tolerance!")