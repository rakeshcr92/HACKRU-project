# summarization_service.py


client = OpenAI(api_key="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")


class SummarizationService:
    def summarize(self, transcript, model_name, api_key, prompt):
        # Using GPT for summarizing transcript
        

        response = client.chat.completions.create(model=model_name,
        temperature=1,
        messages=dialogue)
        return response.choices[0].message.content
