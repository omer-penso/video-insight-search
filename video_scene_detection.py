import os
import cv2
import logging
import json
import moondream as md
from PIL import Image
from math import ceil
from rapidfuzz import fuzz
from dotenv import load_dotenv
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def ensure_output_folder_exists(output_folder):
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            logger.info(f"Output folder created: {output_folder}")
    except Exception as e:
        logger.error(f"Error creating output folder: {e}")
        raise


def open_video_file(video_path):
    try:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video = open_video(video_path)
        fps = video.frame_rate
        return video, fps
    except FileNotFoundError as e:
        logger.error(e)
        raise
    except Exception as e:
        logger.error(f"Error opening video file: {e}")
        raise


def detect_scenes_in_video(video, fps, threshold, min_scene_length):
    try:
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=int(fps * min_scene_length)))
        scene_manager.detect_scenes(video)
        return scene_manager.get_scene_list()
    except Exception as e:
        logger.error(f"Error detecting scenes: {e}")
        raise


def save_scene_images(video_path, scene_list, output_folder):
    try:
        cap = cv2.VideoCapture(video_path)
        for i, (start_time, _) in enumerate(scene_list):
            frame_number = start_time.get_frames()
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if ret:
                scene_image_path = os.path.join(output_folder, f"scene_{i + 1}.jpg")
                try:
                    cv2.imwrite(scene_image_path, frame)
                    logger.info(f"Saved: {scene_image_path}")
                except Exception as e:
                    logger.error(f"Error saving scene image {scene_image_path}: {e}")
            else:
                logger.warning(f"Failed to capture frame for scene {i + 1}.")
        cap.release()
    except Exception as e:
        logger.error(f"Error during frame extraction or saving: {e}")
        raise


def detect_scenes(video_path, output_folder, threshold=9.0, min_scene_length=0.6):
    try:
        ensure_output_folder_exists(output_folder)
        video, fps = open_video_file(video_path)
        scene_list = detect_scenes_in_video(video, fps, threshold, min_scene_length)
        logger.info(f"Detected {len(scene_list)} scenes.")
        save_scene_images(video_path, scene_list, output_folder)
        return len(scene_list)
    except Exception as e:
        logger.error(f"Error in scene detection: {e}")
        return 0


def generate_captions_with_moondream(num_scenes, image_folder, output_json="scene_captions.json"):
    api_key = os.getenv("MOONDREAM_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Make sure MOONDREAM_API_KEY is set in the .env file.")

    model = md.vl(api_key=api_key)
    captions = {}

    # Iterate over each scene image and generate a caption
    for scene_number in range(1, num_scenes + 1):
        try:
            image_path = os.path.join(image_folder, f"scene_{scene_number}.jpg")

            image = Image.open(image_path)
           
            caption = model.caption(image, length="short")["caption"]
            captions[scene_number] = caption
        except Exception as e:
            logger.error(f"Error generating caption for scene_{scene_number}.jpg: {e}")

    # Save all captions to a JSON file
    try:
        with open(output_json, "w") as f:
            json.dump(captions, f, indent=4)
        logger.info(f"Captions saved to {output_json}")
    except Exception as e:
        logger.error(f"Error saving captions to {output_json}: {e}")
        raise

    return output_json


def search_captions(search_word, captions_file, threshold=70):
    """
    Search for scenes containing the input word in the captions using fuzzy matching.
    """
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


def create_collage(matching_scenes, image_folder, output_collage="collage.png"):
    """
    Create a collage of images corresponding to the matching scenes.
    """
    try:
        images = [
            Image.open(os.path.join(image_folder, f"scene_{scene}.jpg"))
            for scene in matching_scenes
            if os.path.exists(os.path.join(image_folder, f"scene_{scene}.jpg"))
        ]

        if not images:
            logger.warning("No valid images found for the given scenes.")
            return None

        img_width, img_height = images[0].size
        cols = min(4, len(images))
        rows = ceil(len(images) / cols)

        # Create a blank canvas for the collage
        collage = Image.new("RGB", (cols * img_width, rows * img_height))

        # Paste images onto the collage
        for idx, img in enumerate(images):
            x = (idx % cols) * img_width
            y = (idx // cols) * img_height
            collage.paste(img, (x, y))

        collage.save(output_collage)
        logger.info(f"Collage saved to {output_collage}")
        collage.show()

        return output_collage
    except Exception as e:
        logger.error(f"Error creating collage: {e}")
        return None 


def extract_unique_words(captions_file):
    """
    Extract unique words from all captions for auto-complete suggestions.
    """
    try:
        with open(captions_file, "r") as f:
            captions = json.load(f)

        # Extract all words from the captions
        words = set()
        for caption in captions.values():
            for word in caption.split():
                words.add(word.strip(",.!?").lower())  

        return sorted(words)
    except Exception as e:
        logger.error(f"Error extracting words from captions: {e}")
        return []


def get_search_word_with_autocomplete(captions_file):
    """
    Prompt the user to input a search word with auto-complete suggestions.
    """
    words = extract_unique_words(captions_file)

    word_completer = WordCompleter(words, ignore_case=True)

    search_word = prompt("Search the video using a word: ", completer=word_completer)

    return search_word.strip()
    
if __name__ == "__main__":
    try:
        load_dotenv()

        video_file = "The_Super_Mario_Trailer.mp4"
        scenes_folder = "scene_image"
        captions_file = "scene_captions.json"
        collage_file = "collage.png"

        # Check if JSON file exists
        if os.path.exists(captions_file):
            logger.info(f"Captions file '{captions_file}' already exists. Skipping all processing.")
            print(f"Captions already exist in '{captions_file}'.")
        else:
            if not video_file or not scenes_folder:
                raise ValueError("Both video file path and output folder path must be provided.")

            num_scenes = detect_scenes(video_file, output_folder=scenes_folder)
            if num_scenes > 0:
                print(f"Saved images for {num_scenes} scenes in '{scenes_folder}'.")
                generate_captions_with_moondream(num_scenes, scenes_folder, captions_file)
                print(f"Captions generated and saved in '{captions_file}'.")
            else:
                print("No scenes were detected or saved due to errors.")
                raise RuntimeError("No scenes detected or saved.")          
        
        # Prompt user for a search word
        search_word = get_search_word_with_autocomplete(captions_file)

        matching_scenes = search_captions(search_word, captions_file)
        if not matching_scenes:
            print(f"No scenes found for the word '{search_word}'.")
            logger.info(f"No scenes found for the word '{search_word}'.")
        else:
            print(f"Scenes found for the word '{search_word}': {matching_scenes}")
            logger.info(f"Scenes found for the word '{search_word}': {matching_scenes}")

            collage_path = create_collage(matching_scenes, scenes_folder, collage_file)
            if collage_path:
                print(f"Collage created and saved as '{collage_path}'.")
            else:
                print("Failed to create collage.")
    except Exception as e:
        logger.critical(f"Unexpected error: {e}")
        print("A critical error occurred. Please check the logs for more details.")
