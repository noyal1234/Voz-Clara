
""" Using UForm Gen2 Qwen 500M Vision Model with Text-to-Speech """


import cv2
import os
import time
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor
from gtts import gTTS
import pygame

# Constants
VISION_MODEL_ID = "unum-cloud/uform-gen2-qwen-500m"
TAKE_PHOTO_COMMAND = "Take a photo."
OUTPUT_DESCRIPTION_FILE = "descriptions.txt"  # Adjust path as needed

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Load vision model and processor on CPU
print('Loading models...')
vision_model = AutoModel.from_pretrained(VISION_MODEL_ID, trust_remote_code=True)
# Convert model to float32 to match input dtype
vision_model = vision_model.float()
processor = AutoProcessor.from_pretrained(VISION_MODEL_ID, trust_remote_code=True)

def speak_text(text):
    """
    Convert text to speech using Google TTS and play it.
    
    Args:
        text (str): The text to be spoken.
    """
    tts = gTTS(text=text, lang='en')
    temp_file = "temp_speech.mp3"
    tts.save(temp_file)
    
    # Play the audio file
    pygame.mixer.music.load(temp_file)
    pygame.mixer.music.play()
    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    
    # Delete the temporary file immediately after playing
    try:
        os.remove(temp_file)
        print("Audio file deleted")
    except Exception as e:
        print(f"Error deleting audio file: {e}")

def take_picture(number):
    """
    Capture a picture using the connected camera and save it as a JPEG file.
    
    Args:
        number (int): Used to name the saved photo.
        
    Returns:
        str: The filename of the captured image or None if capture fails.
    """
    camera_id = "/dev/video0"
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    # Set resolution as needed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            filename = f'photo_{number}.jpg'
            cv2.imwrite(filename, frame)
            cap.release()
            print(f"Photo captured: {filename}")
            return filename
        else:
            cap.release()
            print("Failed to capture image from camera.")
            return None
    else:
        print("Unable to open camera.")
        return None

def main():
    photo_number = 1
    print("Control Options:")
    print("   t - Take a new photo")
    print("   y - Ask a question")
    print("   q - Quit")
    
    # Start by capturing an initial image
    filename = take_picture(photo_number)
    if filename is None:
        print("No image captured. Exiting.")
        return

    image = Image.open(filename)
    
    while True:
        option = input("\nEnter your option (t/y/q): ").strip().lower()
        
        if option == "q":
            print("Exiting...")
            break

        elif option == "t":
            photo_number += 1
            new_filename = take_picture(photo_number)
            if new_filename is not None:
                filename = new_filename
                image = Image.open(filename)
            else:
                print("Photo capture failed.")

        elif option == "y":
            question = input("Enter your question or instruction: ").strip()
            if question == TAKE_PHOTO_COMMAND:
                print("Simulating photo capture...")
                photo_number += 1
                new_filename = take_picture(photo_number)
                if new_filename is not None:
                    filename = new_filename
                    image = Image.open(filename)
                else:
                    print("Photo capture failed.")
            else:
                # Process the text and image with the vision model
                inputs = processor(text=[question], images=[image], return_tensors="pt")
                print("Generating output...")
                with torch.inference_mode():
                    output = vision_model.generate(
                        **inputs,
                        do_sample=False,
                        use_cache=True,
                        max_new_tokens=256,
                        eos_token_id=151645,
                        pad_token_id=processor.tokenizer.pad_token_id
                    )
                prompt_len = inputs["input_ids"].shape[1]
                decoded_text = processor.batch_decode(output[:, prompt_len:])[0]
                # Clean up any trailing special tokens
                text = decoded_text.replace("<|im_end|>", "").strip()
                print(f"Response: {text}")

                # Save response to file.
                with open(OUTPUT_DESCRIPTION_FILE, 'a') as f:
                    f.write(text + "\n")

                # Use Google TTS to speak out the response
                speak_text(text)

        else:
            print("Invalid option. Please enter t, y, or q.")
        # Optional: short delay between iterations
        time.sleep(0.5)

if __name__ == "_main_":
    main()