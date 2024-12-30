from t5_architecture import T5_BASE_GENERATION, GenerationUtils

# Initialize the model and transformation
padding_idx = 0
eos_idx = 1
max_seq_len = 512

t5_base = T5_BASE_GENERATION
transform = t5_base.transform()
model = t5_base.get_model()
model.eval()

sequence_generator = GenerationUtils(model)

# Define examples for different tasks
def summarize_example():
    task = "summarize"
    input_text = [
        "summarize: Concerns about OpenAI's business model are raised, with suggestions to emulate Apple's app platform approach by allowing developers to publish AI applications and take a percentage cut. Users also point out the absence of an app platform for exploring ChatGPT-based applications as a potential revenue stream."
    ]

    model_input = transform(input_text)
    model_output = sequence_generator.generate(model_input, eos_idx=eos_idx, num_beams=1)
    output_text = transform.decode(model_output.tolist())

    print("Summarization Task:")
    for i, text in enumerate(output_text):
        print(f"Example {i+1}:")
        print(f"Input: {input_text[i]}")
        print(f"Output: {text}\n")


def sentiment_example():
    task = "sst2 sentence"
    input_text = ["sst2 sentence: His behavior struck me as odd. Be careful"]

    model_input = transform(input_text)
    model_output = sequence_generator.generate(model_input, eos_idx=eos_idx, num_beams=1)
    output_text = transform.decode(model_output.tolist())

    print("Sentiment Analysis Task:")
    for i, text in enumerate(output_text):
        print(f"Example {i+1}:")
        print(f"Input: {input_text[i]}")
        print(f"Output: {text}\n")


def translation_example():
    task = "translate English to German"
    input_text = ["translate English to German: The cat jumped on the hat"]

    model_input = transform(input_text)
    model_output = sequence_generator.generate(model_input, eos_idx=eos_idx, num_beams=1)
    output_text = transform.decode(model_output.tolist())

    print("Translation Task:")
    for i, text in enumerate(output_text):
        print(f"Example {i+1}:")
        print(f"Input: {input_text[i]}")
        print(f"Output: {text}\n")

# Run the examples
summarize_example()
sentiment_example()
translation_example()
