from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=""
)

with open('./list', 'r', encoding='utf-8') as file:
    semantic_text_all = file.readlines()
    for classname in semantic_text_all:
        prompt = f"What does a {classname} look like? Please force answer in the format: A {classname} has A, B, C,..., where A, B, and C are noun phrases describing a {classname}."
        completion = client.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
            messages=[{"role": "user",
                       "content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=77,
            stream=False
        )

    message_content = completion.choices[0].message.content
    print(message_content)
