
""" Use the Gemini API to assist blind users by answering questions about images """


import cv2
import os
import time
import torch
import re
import pygame
from PIL import Image
from gtts import gTTS
import gc

import google.generativeai as genai

# Constants
OUTPUT_DESCRIPTION_FILE = "descriptions.txt"
TAKE_PHOTO_COMMAND = "Take a photo."
GEMINI_API_KEY = "API_KEY"  # Your Gemini API key

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Initialize Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def clean_answer(raw_answer):
    """Clean model output to get just the answer part."""
    if "Answer:" in raw_answer:
        parts = raw_answer.split("Answer:")
        cleaned_answer = parts[-1].strip()
    else:
        pattern = r"<image>\s*Question:.*?Answer:"
        cleaned_answer = re.sub(pattern, "", raw_answer, flags=re.DOTALL).strip()

    # Remove bold markdown
    cleaned_answer = cleaned_answer.replace("", "")

    return cleaned_answer

def format_as_guide(answer):
    """Format the answer to sound like a helpful guide for a blind person."""
    if not answer or len(answer) < 5 or "i can't" in answer.lower() or "cannot" in answer.lower():
        return "I'm sorry, I can't provide a clear description of what's in this image. Would you like me to try a different approach or help with something else?"

    formatted_answer = f"I can see that {answer}. Would you like me to describe any specific part in more detail?"

    return formatted_answer

def speak_text(text):
    """Convert text to speech using Google TTS and play it."""
    # Use a lower quality for faster processing and less memory
    tts = gTTS(text=text, lang='en', slow=False)
    temp_file = "temp_speech.mp3"
    tts.save(temp_file)

    pygame.mixer.music.load(temp_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    try:
        os.remove(temp_file)
    except Exception as e:
        print(f"Error deleting audio file: {e}")

def take_picture(number):
    """Capture a picture using memory-efficient settings."""
    camera_id = "/dev/video0"
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

    # Lower resolution to save memory
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced from 1280
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduced from 720

    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()  # Release camera immediately

        if ret:
            filename = f'photo_{number}.jpg'
            # Use lower JPEG quality for smaller file size
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"Photo captured: {filename}")
            return filename
        else:
            print("Failed to capture image from camera.")
            return None
    else:
        print("Unable to open camera.")
        return None

def answer_question_with_gemini(image_path, question):
    """Process an image and question using the Gemini API."""
    try:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
        prompt = "You are a vision assistant for a blind person. Please provide short and clear answers to the questions based on the image. Answer as concisely as possible."
        question = f"{prompt} Question: {question}"
        contents = [
            {"mime_type": "image/jpeg", "data": image_data},
            question
        ]

        response = model.generate_content(contents)
        response.resolve()
        answer = response.text

        if answer is None or answer.strip() == "":
            return "I'm unable to answer that question based on the image."
        else:
            
            guide_answer = clean_answer(answer)
            return guide_answer

    except Exception as e:
        print(f"Error processing image with Gemini: {str(e)}")
        return "I encountered an error while trying to analyze the image."

def main():
    # Helper questions for blind assistance
    helper_questions = [
        "What's in this image?",
        "Are there any obstacles in front of me?",
        "Are there people nearby?",
        "Is there text I should know about?",
        "What's the general environment like?"
    ]

    try:
        print("\nGemini Vision Assistant is ready!")
        print("Control Options:")
        print("    t - Take a new photo")
        print("    q - Ask a question about the current photo")
        print("    h - Use a suggested helper question")
        print("    x - Exit")
        speak_text("Vision Assistant ready. Press t to take a photo.")

        photo_number = 0
        filename = None

        while True:
            option = input("\nEnter your option (t/q/h/x): ").strip().lower()

            if option == "x":
                print("Exiting...")
                speak_text("Shutting down vision assistant. Goodbye!")
                break

            elif option == "t":
                speak_text("Taking a new photo.")
                photo_number += 1
                # Delete previous photo if it exists
                if filename and os.path.exists(filename):
                    try:
                        os.remove(filename)
                    except:
                        pass

                new_filename = take_picture(photo_number)
                if new_filename is not None:
                    filename = new_filename
                    speak_text("New photo captured.")
                else:
                    speak_text("Photo capture failed.")

            elif option == "q" or option == "h":
                if not filename or not os.path.exists(filename):
                    speak_text("Please take a photo first.")
                    continue

                if option == "q":
                    question = input("Enter your question about the image: ").strip()
                elif option == "h":
                    print("\nSuggested helper questions:")
                    for i, q in enumerate(helper_questions, 1):
                        print(f"{i}. {q}")

                    q_choice = input("Enter question number (or 'r' to return): ").strip()
                    if q_choice.lower() == 'r':
                        continue

                    try:
                        q_num = int(q_choice)
                        if 1 <= q_num <= len(helper_questions):
                            question = helper_questions[q_num - 1]
                            print(f"Selected: {question}")
                        else:
                            print("Invalid question number.")
                            speak_text("Invalid question number selected.")
                            continue
                    except ValueError:
                        print("Please enter a valid number.")
                        speak_text("Invalid input. Please try again.")
                        continue

                print("Processing question...")
                speak_text("Analyzing the image...")

                try:
                    response = answer_question_with_gemini(filename, question)
                    print(f"Response: {response}")

                    # Save response to file
                    with open(OUTPUT_DESCRIPTION_FILE, 'a') as f:
                        f.write(f"Q: {question}\nA: {response}\n\n")

                    # Speak the response
                    speak_text(response)
                except Exception as e:
                    error_msg = f"Error processing question: {str(e)}"
                    print(error_msg)
                    speak_text("I had trouble processing that image. Let's try taking a new photo.")

            else:
                print("Invalid option. Please enter t, q, h, or x.")
                speak_text("Invalid option. Please try again.")

    except Exception as e:
        error_msg = f"Error initializing the application: {str(e)}"
        print(error_msg)
        speak_text("An error occurred. The application must close.")

    finally:
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if _name_ == "_main_":
    main()