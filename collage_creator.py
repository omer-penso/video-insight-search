import os
import logging
from PIL import Image
from math import ceil

logger = logging.getLogger(__name__)

def create_collage(matching_scenes, image_folder, output_collage="collage.png"):
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

        collage = Image.new("RGB", (cols * img_width, rows * img_height))

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
