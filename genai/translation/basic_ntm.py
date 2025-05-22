# Imports the Google Cloud Translation library
from utils.time_utils import timing_decorator
from typing import List
from google.cloud import translate


# Initialize Translation client
@timing_decorator
def translate_text(
    project_id: str, strs_to_translate: List[str]
) -> translate.TranslationServiceClient:
    """Translating Text."""

    client = translate.TranslationServiceClient()

    location = "us-central1"

    parent = f"projects/{project_id}/locations/{location}"

    # Translate text from fr to ja
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": strs_to_translate,
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": "fr",
            "target_language_code": "zh",
        }
    )

    # Display the translation for each input text provided
    for translation in response.translations:
        print(f"Translated text: {translation.translated_text}")

    return response

text1 = """Cloud Translation API utilise la technologie de traduction automatique neuronale de Google pour vous permettre de traduire dynamiquement du texte via l'API à l'aide d'un modèle personnalisé pré-entraîné de Google ou d'un modèle de langage étendu (LLM) spécialisé en traduction."""
contents = [text1]
translate_text("ali-icbu-gpu-project", contents)