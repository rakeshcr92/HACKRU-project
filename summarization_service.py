# summarization_service.py

from openai import OpenAI

client = OpenAI(api_key="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")


class SummarizationService:
    def summarize(self, transcript, model_name, api_key, prompt):
        # Summarize transcript using GPT
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript}
        ]
        response = client.chat.completions.create(model=model_name,
        temperature=1,
        messages=messages)
        return response.choices[0].message.content
