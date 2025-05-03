import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import base64
import os
from openai import OpenAI
from dotenv import load_dotenv
import logging
import warnings
import threading
import time
from queue import Queue

logging.getLogger("ultralytics").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

detection_queue = Queue()

def play_audio(file_path):
    try:
        os.system(f'play {file_path} -q')
    except Exception as e:
        print(f"Error playing audio: {e}")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def process_detection():
    """Background thread to process detections"""
    last_announcement = {}
    announcement_cooldown = 5  # seconds
    
    while True:
        try:
            detection_data = detection_queue.get()
            if detection_data is None:  # Exit signal
                break
                
            frame, class_name, confidence = detection_data
            current_time = time.time()
            
            if (class_name not in last_announcement or 
                current_time - last_announcement[class_name] > announcement_cooldown):
                
                temp_image_path = f"temp_frame_{threading.get_ident()}.jpg"
                cv2.imwrite(temp_image_path, frame)
                
                try:
                    base64_image = encode_image(temp_image_path)
                    
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"There is a {class_name} detected with {confidence:.2f} confidence. In one succinct sentence say: Start your sentence with a warning (ex Be proceed with caution... or Be careful...). with a confidence of {confidence}, you are approaching a {class_name}, it is (location relative to you)"
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        }
                                    }
                                ]
                            }
                        ]
                    )
                    
                    response_text = response.choices[0].message.content
                    print(f"Analysis: {response_text}")
                    
                    speech_file_path = f"speech_{threading.get_ident()}.mp3"
                    speech_response = client.audio.speech.create(
                        model="tts-1",
                        voice=os.getenv('VOICE_TYPE', 'alloy'),
                        input=response_text
                    )
                    speech_response.stream_to_file(speech_file_path)
                    
                    threading.Thread(target=play_audio, args=(speech_file_path,), daemon=True).start()
                    
                    last_announcement[class_name] = current_time
                    
                except Exception as e:
                    print(f"Processing error: {e}")
                finally:
                    # Cleanup
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)
                    if os.path.exists(speech_file_path):
                        time.sleep(1)
                        try:
                            os.remove(speech_file_path)
                        except:
                            pass
                            
        except Exception as e:
            print(f"Detection processing error: {e}")
        finally:
            detection_queue.task_done()

def create_capture():
    max_retries = 20
    retry_delay = 0
    
    for attempt in range(max_retries):
        try:
            cap = cv2.VideoCapture('tcp://192.168.233.4:8000')
            if cap.isOpened():
                print("Successfully connected to camera stream")
                return cap
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
        
        print(f"Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)
    
    raise ConnectionError("Failed to connect to camera stream after multiple attempts")

def main():
    model = YOLO("navigoose.pt", verbose=False)
    
    processing_thread = threading.Thread(target=process_detection, daemon=True)
    processing_thread.start()
    
    while True:
        try:
            cap = create_capture()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to receive frame, attempting to reconnect...")
                    break
                
                results = model.track(frame, persist=True, verbose=False)
                
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    
                    for box in boxes:
                        confidence = float(box.conf)
                        
                        if confidence > 0.50:
                            class_id = int(box.cls)
                            class_name = results[0].names[class_id]
                            
                            try:
                                detection_queue.put_nowait((frame.copy(), class_name, confidence))
                            except:
                                pass  # Queue full, skip this detection
                    
                    annotated_frame = results[0].plot()
                    cv2.imshow('obstacle detection', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt
                    
        except KeyboardInterrupt:
            print("Shutting down...")
            break
        except Exception as e:
            print(f"Error: {e}")
        finally:
            if 'cap' in locals() and cap is not None:
                cap.release()
    
    detection_queue.put(None)  # Signal processing thread to exit
    processing_thread.join(timeout=1)  # Wait for processing thread
    cv2.destroyAllWindows()
    
    for file in os.listdir():
        if file.startswith('temp_frame_') or file.startswith('speech_'):
            try:
                os.remove(file)
            except:
                pass

if __name__ == "__main__":
    main()
