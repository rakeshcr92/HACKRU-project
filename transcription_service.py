# transcription_service.py

import whisper

class TranscriptionService:
    def transcribe(self, link, model_name, api_key):
        # Utilize Whisper for audio transcription.
        model = whisper.load_model(model_name)
        result = model.transcribe(audio=link, verbose=True)
        return result["text"]
