from dotenv import load_dotenv
import logging
import os
from video_processing import detect_scenes
from scenes_caption_generation import (
    generate_captions_with_moondream,
    search_captions,
    get_search_word_with_autocomplete
)
from collage_creator import create_collage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        load_dotenv()

        video_file = "The_Super_Mario_Trailer.mp4"
        scenes_folder = "scene_image"
        captions_file = "scene_captions.json"
        collage_file = "collage.png"

        if not video_file or not scenes_folder:
            raise ValueError("Both video file path and output folder path must be provided.")

        if not os.path.exists(captions_file):
            num_scenes = detect_scenes(video_file, scenes_folder)
            if num_scenes > 0:
                generate_captions_with_moondream(num_scenes, scenes_folder, captions_file)

        search_word = get_search_word_with_autocomplete(captions_file)
        matching_scenes = search_captions(search_word, captions_file)
        if matching_scenes:
            create_collage(matching_scenes, scenes_folder, collage_file)
        else:
            print(f"No scenes found for the word '{search_word}'.")

    except Exception as e:
        logger.critical(f"Unexpected error: {e}")
