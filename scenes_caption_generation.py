import os
import json
import logging
import moondream as md
from PIL import Image
from rapidfuzz import fuzz
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

logger = logging.getLogger(__name__)

def generate_captions_with_moondream(num_scenes, image_folder, output_json="scene_captions.json"):
    api_key = os.getenv("MOONDREAM_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Make sure MOONDREAM_API_KEY is set in the .env file.")

    model = md.vl(api_key=api_key)
    captions = {}

    for scene_number in range(1, num_scenes + 1):
        try:
            image_path = os.path.join(image_folder, f"scene_{scene_number}.jpg")
            image = Image.open(image_path)
            caption = model.caption(image, length="short")["caption"]
            captions[scene_number] = caption
        except Exception as e:
            logger.error(f"Error generating caption for scene_{scene_number}.jpg: {e}")

    try:
        with open(output_json, "w") as f:
            json.dump(captions, f, indent=4)
        logger.info(f"Captions saved to {output_json}")
    except Exception as e:
        logger.error(f"Error saving captions to {output_json}: {e}")
        raise

    return output_json

def extract_unique_words(captions_file):
    try:
        with open(captions_file, "r") as f:
            captions = json.load(f)

        words = set()
        for caption in captions.values():
            for word in caption.split():
                words.add(word.strip(",.!?").lower())

        return sorted(words)
    except Exception as e:
        logger.error(f"Error extracting words from captions: {e}")
        return []

def search_captions(search_word, captions_file, threshold=70):
    try:
        with open(captions_file, "r") as f:
            captions = json.load(f)

        matching_scenes = []
        for scene_number, caption in captions.items():
            match_score = fuzz.partial_ratio(search_word.lower(), caption.lower())
            if match_score >= threshold:
                matching_scenes.append(int(scene_number))

        return matching_scenes
    except Exception as e:
        logger.error(f"Error searching captions: {e}")
        return []

def get_search_word_with_autocomplete(captions_file):
    """
    Prompt the user to input a search word with auto-complete suggestions.
    """
    words = extract_unique_words(captions_file)
    word_completer = WordCompleter(words, ignore_case=True)
    search_word = prompt("Search the video using a word: ", completer=word_completer)
    return search_word.strip()
