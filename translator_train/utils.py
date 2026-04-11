def format_to_chat(example):
    system_content = "You are a professional English to Moroccan Darija translator. Translate the given English text accurately into Moroccan Darija written in Arabic script, maintaining the original meaning, tone, and context."
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": example["english"]},
        {"role": "assistant", "content": example["darija"]},
    ]
    return {"messages": messages}
