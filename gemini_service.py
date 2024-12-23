import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_scene_start_times_gemini(user_query, video_file):
    """Retrieve scene start times from Gemini API."""
    try:
        gemini_api = os.getenv('GEMINI_KEY')
        if not gemini_api:
            raise ValueError("Gemini API key not found. Ensure GEMINI_KEY is set in the .env file.")

        genai.configure(api_key=gemini_api)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""Analyze the uploaded video and identify the specific timestamps for scenes that match the provided Query.
            For each identified scene, provide:
            The start time in the format S.mmm.
            Return the result as a valid JSON array with NO additional text, markers, or formatting. 
            Example of the desired response: [0.0, 2.002, 7.925]
            Query: {user_query}
        """

        response = model.generate_content( 
                    [video_file, prompt],
                    generation_config=genai.GenerationConfig(
                        temperature=0.3
                    )
        )
        response_text = response.text.strip()
        response_text = response_text.strip("'''").strip("```").strip("json")

        return response_text
    except Exception as e:
        logger.error(f"Error retrieving scene start times from Gemini: {e}")
        raise


#testing the module
if __name__ == "__main__":
    load_dotenv()  

    user_query = input("Enter your search query for Gemini: ")
    video_file = "The_Super_Mario_Trailer.mp4"  

    try:
        start_times = get_scene_start_times_gemini(user_query, video_file)
        print(f"Start times retrieved: {start_times}")
    except Exception as e:
        print(f"An error occurred: {e}")