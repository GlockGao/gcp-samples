import os
import sys
from typing import List

# Import the Google Cloud Translation library.
from google.cloud import translate_v3 as translate


# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将 utils 目录的父目录（即 your_project 目录）添加到 sys.path
sys.path.append(os.path.join(current_dir, '..')) # 或者直接 sys.path.append(current_dir) 如果 utils 是顶层包

from utils.time_utils import timing_decorator


@timing_decorator
def translate_text(
    project_id: str,
    text: List[str],
    source_language_code: str = "en-US",
    target_language_code: str = "fr",
) -> translate.TranslationServiceClient:
    """Translate Text from a Source language to a Target language.
    Args:
        project_id: GCP project id.
        text: The content to translate.
        source_language_code: The code of the source language.
        target_language_code: The code of the target language.
            For example: "fr" for French, "es" for Spanish, etc.
            Find available languages and codes here:
            https://cloud.google.com/translate/docs/languages#neural_machine_translation_model
    """

    # Initialize Translation client.
    client = translate.TranslationServiceClient()
    parent = f"projects/{project_id}/locations/global"

    # MIME type of the content to translate.
    # Supported MIME types:
    # https://cloud.google.com/translate/docs/supported-formats
    mime_type = "text/plain"

    # Translate text from the source to the target language.
    response = client.translate_text(
        contents=text,
        parent=parent,
        mime_type=mime_type,
        source_language_code=source_language_code,
        target_language_code=target_language_code,
    )

    # Display the translation for the text.
    # For example, for "Hello! How are you doing today?":
    # Translated text: Bonjour comment vas-tu aujourd'hui?
    for translation in response.translations:
        print(f"Translated text: {translation.translated_text}")

    return response


text1 = """Cloud Translation API utilise la technologie de traduction automatique neuronale de Google pour vous permettre de traduire dynamiquement du texte via l'API à l'aide d'un modèle personnalisé pré-entraîné de Google ou d'un modèle de langage étendu (LLM) spécialisé en traduction."""
contents = [text1]
translate_text(project_id="ali-icbu-gpu-project", 
               text=contents, 
               source_language_code="en-US",
               target_language_code="zh")