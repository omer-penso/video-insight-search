import os
import cv2
import logging
import json
import moondream as md
from PIL import Image
from dotenv import load_dotenv
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

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


if __name__ == "__main__":
    try:
        load_dotenv()

        video_file = "The_Super_Mario_Trailer.mp4"
        scenes_folder = "scene_image"
        captions_file = "scene_captions.json"

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
    except Exception as e:
        logger.critical(f"Unexpected error: {e}")
        print("A critical error occurred. Please check the logs for more details.")
