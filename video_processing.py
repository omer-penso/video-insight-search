import os
import cv2
import logging
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

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
          # Extract the full scene list (start_time, end_time)
        full_scene_list = scene_manager.get_scene_list()

        # Extract only the start times as seconds (float)
        scenes_start_time_list = [start_time.get_seconds() for start_time, _ in full_scene_list]

        return scenes_start_time_list
    except Exception as e:
        logger.error(f"Error detecting scenes: {e}")
        raise
# to add that he will also get fps and then will go over the scene list and mult it to get the frame number
def save_scene_images(video_path, fps, scenes_start_time_list, output_folder):
    try:
        cap = cv2.VideoCapture(video_path)
        for i, start_time in enumerate(scenes_start_time_list):
            frame_number = fps * start_time
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
        save_scene_images(video_path, fps, scene_list, output_folder)
        return len(scene_list)
    except Exception as e:
        logger.error(f"Error in scene detection: {e}")
        return 0
