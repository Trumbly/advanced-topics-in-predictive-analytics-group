import openai
import yaml

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required by SDK but unused by Ollama
)

def call_llm(messages: list[dict], temperature: float = None) -> str:
    temp = temperature if temperature is not None else cfg["agent"]["temperature"]
    response = client.chat.completions.create(
        model=cfg["agent"]["model"],
        messages=messages,
        temperature=temp,
    )
    return response.choices[0].message.content.strip()
