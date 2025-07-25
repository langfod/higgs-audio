import requests
import argparse
import os
import random
import time
import json

# --- Helper Functions ---

def upload_file_to_gradio(session, host, file_path):
    """Uploads a file to the Gradio /upload endpoint."""
    print(f"--- Step 1: Uploading Reference Audio ---")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reference audio file not found: {file_path}")

    url = f"{host}/gradio_api/upload"
    files = {'files': (os.path.basename(file_path), open(file_path, 'rb'), 'audio/wav')}
    
    print(f"Uploading {os.path.basename(file_path)} to {url}...")
    try:
        response = session.post(url, files=files, timeout=30)
        response.raise_for_status()
        server_path = response.json()[0]
        print(f"Upload successful. Server path: {server_path}\n")
        return server_path
    except requests.exceptions.RequestException as e:
        print(f"Error uploading file: {e}")
        return None

def build_tts_payload(model, text, ref_audio_server_path, mood, seed):
    """Builds the JSON payload for the /generate_audio call."""
    print("--- Step 2: Building TTS Request Payload ---")
    print(f"Using model: {model}")
    print(f"Using random seed: {seed}")

    payload = {
        "data": [
            model, text, "en-us",
            { "meta": {"_type": "gradio.FileData"}, "path": ref_audio_server_path },
            None,
            0.8 if mood == 'HAPPY' else 0.05, 0.8 if mood == 'SAD' else 0.05,
            0.8 if mood == 'DISGUSTED' else 0.05, 0.8 if mood == 'AFRAID' else 0.05,
            0.8 if mood == 'SURPRISED' else 0.05, 0.8 if mood == 'ANGRY' else 0.05,
            0.05, 0.8 if mood == 'NEUTRAL' else 0.2,
            0.7, 24000.0, 45.0, 14.6, 4.0, True, 3.0, 0.9, 1, 0.2,
            False, 0.7, False, seed, False, []
        ]
    }
    print(f"Payload constructed for mood: {mood} with {len(payload['data'])} parameters.\n")
    return payload

def poll_for_result(session, host, event_id):
    """Polls the SSE endpoint for the final result."""
    timeout = 300 
    event_url = f"{host}/gradio_api/call/generate_audio/{event_id}"
    print(f"Job submitted. Event ID: {event_id}. Waiting for result (timeout: {timeout}s)...")
    
    start_time = time.time()
    try:
        with session.get(event_url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if time.time() - start_time > timeout:
                    print("Error: Polling timed out.")
                    return None
                    
                if line:
                    decoded_line = line.decode('utf-8')
                    print(f"[SERVER SENT]: {decoded_line}")
                    
                    if decoded_line.startswith('data:'):
                        try:
                            content = decoded_line[5:].strip()
                            if not content:
                                continue
                                
                            event_data = json.loads(content)
                            
                            # --- FIX #1: ADAPT TO THE ACTUAL SUCCESS PAYLOAD ---
                            # Instead of looking for {"msg": "process_completed"}, we look for the
                            # actual data structure of a successful response: a list where the
                            # first item is a dictionary containing a 'path'.
                            if (isinstance(event_data, list) and
                                len(event_data) > 0 and
                                isinstance(event_data[0], dict) and
                                'path' in event_data[0]):
                                print(">>> Found successful result payload.")
                                return event_data # Return the successful list payload
                                
                        except json.JSONDecodeError:
                            pass

    except requests.exceptions.ReadTimeout:
        print(f"Error: Read timeout after {timeout} seconds. The server took too long to send data.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error while polling for result: {e}")
        return None

    print("Error: Stream ended without a successful result payload.")
    return None

def generate_tts_audio(host, model, ref_audio_path, text, output_path, mood):
    """Main function to handle the full TTS generation process."""
    
    with requests.Session() as session:
        ref_audio_server_path = upload_file_to_gradio(session, host, ref_audio_path)
        if not ref_audio_server_path:
            return

        seed = random.randint(1, 2**32 - 1)
        payload = build_tts_payload(model, text, ref_audio_server_path, mood, seed)

        print("--- Step 3: Submitting TTS Job and Fetching Result ---")
        submit_url = f"{host}/gradio_api/call/generate_audio"
        try:
            response = session.post(submit_url, json=payload, timeout=30)
            response.raise_for_status()
            event_id = response.json().get("event_id")
            if not event_id:
                print("Error: Did not receive an event_id from the server.")
                print(f"Server Response: {response.text}")
                return
        except requests.exceptions.RequestException as e:
            print(f"Error submitting TTS job: {e}")
            return

        result_json = poll_for_result(session, host, event_id)
        
        if not result_json:
            print("Failed to get a final result from the server.")
            return

        try:
            # --- FIX #2: PARSE THE NEW RESULT STRUCTURE ---
            # result_json is now the list itself, e.g., [{"path": ...}, 12345]
            # The dictionary we need is the first element of that list.
            output_data = result_json[0]
            audio_server_path = output_data.get('path')
            
            if not audio_server_path:
                print("Error: Final result did not contain an audio path.")
                print(f"Full result: {result_json}")
                return

            print(f"\n--- Step 4: Downloading Result ---")
            audio_url = f"{host}/gradio_api/file={audio_server_path}"
            print(f"Downloading audio from {audio_url}...")
            
            audio_response = session.get(audio_url, timeout=60)
            audio_response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(audio_response.content)
            
            print(f"Successfully saved generated audio to {output_path}")

        except (KeyError, IndexError, TypeError) as e:
            print(f"Error parsing the final result JSON: {e}")
            print("The structure of the server's response may have changed.")
            print(f"Full result received: {json.dumps(result_json, indent=2)}")

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for Higgs/Zonos TTS API.")
#    parser.add_argument('--host', type=str, default="http://185.62.108.226:55668", help="Host URL of the Gradio server.")
    parser.add_argument('--host', type=str, default="http://localhost:7860", help="Host URL of the Gradio server.")
    parser.add_argument('--model', type=str, default="Zyphra/Zonos-v0.1-hybrid", 
                        choices=['Zyphra/Zonos-v0.1-transformer', 'Zyphra/Zonos-v0.1-hybrid'],
                        help="The TTS model to use.")
    parser.add_argument('--ref-audio', type=str, required=True, help="Path to the reference WAV file.")
    parser.add_argument('--text', type=str, required=True, help="Text to synthesize.")
    parser.add_argument('--output', type=str, default="output.wav", help="Path to save the generated audio.")
    parser.add_argument('--mood', type=str, default="NEUTRAL", choices=['NEUTRAL', 'HAPPY', 'SAD', 'ANGRY', 'AFRAID', 'SURPRISED', 'DISGUSTED'], help="Emotion for the TTS.")
    
    args = parser.parse_args()
    
    generate_tts_audio(
        host=args.host, 
        model=args.model,
        ref_audio_path=args.ref_audio, 
        text=args.text, 
        output_path=args.output, 
        mood=args.mood.upper()
    )
