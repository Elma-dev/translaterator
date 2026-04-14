from configs import OUTPUT_DIR
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def translate_to_darija(text, model, tokenizer):
    system_prompt = "You are a professional English to Moroccan Darija translator. Translate the given English text accurately into Moroccan Darija written in Arabic script, maintaining the original meaning, tone, and context."

    # Format according to the chat template used in training
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    # Apply the chat template
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    # Generate translation
    outputs = model.generate(
        input_ids, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9
    )

    # Decode only the generated part
    prediction = tokenizer.decode(
        outputs[0][input_ids.shape[-1] :], skip_special_tokens=True
    )
    return prediction.strip()


if __name__ == "__main__":
    model_path = OUTPUT_DIR
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Test Sentences
    test_sentences = [
        "How are you doing today?",
        "I want to go to the market to buy some fresh bread.",
        "Could you please tell me the way to the train station?",
        "The food in this restaurant is very delicious but a bit expensive.",
        "I am so happy to see you again after all this time.",
    ]

    print("\n--- Translation Results ---\n")
    for sentence in test_sentences:
        translation = translate_to_darija(sentence, model, tokenizer)
        print(f"English: {sentence}")
        print(f"Darija : {translation}\n")
