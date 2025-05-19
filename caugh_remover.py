clip_path = "/Users/ari/Movies/2025-05-17 19-32-05.mov"

import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import requests # For fetching the class map

def detect_cough_in_audio(audio_file_path=clip_path):
    """
    Detects cough and throat clearing sounds in an audio file using YAMNet.
    """
    print("Loading YAMNet model...")
    try:
        # Load YAMNet model from TensorFlow Hub
        # YAMNet is a pre-trained deep net that predicts 521 audio event classes.
        model = hub.load('https://tfhub.dev/google/yamnet/1')
    except Exception as e:
        print(f"Error loading YAMNet model: {e}")
        return

    print(f"Loading audio file: {audio_file_path}...")
    try:
        # Load audio file using librosa
        # YAMNet expects 16 kHz, mono audio. librosa.load handles resampling and mono conversion by default.
        # It also normalizes the audio to the range [-1.0, 1.0].
        waveform, sr = librosa.load(audio_file_path, sr=16000, mono=True)
    except FileNotFoundError:
        print(f"Error: Audio file not found at {audio_file_path}")
        return
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    if waveform.size == 0:
        print("Error: Audio file is empty.")
        return

    print("Running inference with YAMNet...")
    try:
        # Perform inference. The model returns scores, embeddings, and the log mel spectrogram.
        # Scores are per-frame predictions for 521 classes.
        # Waveform needs to be a 1D float32 Tensor or NumPy array.
        scores, embeddings, spectrogram = model(waveform)
    except Exception as e:
        print(f"Error during model inference: {e}")
        return

    # Scores are logits (uncalibrated log-odds).
    # To get probabilities, you can apply tf.sigmoid(scores).

    print("Loading class labels...")
    labels_csv_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
    try:
        response = requests.get(labels_csv_url)
        response.raise_for_status() # Raise an exception for HTTP errors
        labels_raw = response.text
        # The CSV has columns: index, mid, display_name
        # We need the display_name. Skip the header row.
        class_names = [line.split(',')[2].strip().strip('"') for line in labels_raw.splitlines()][1:]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching class labels: {e}")
        return
    except Exception as e:
        print(f"Error parsing class labels: {e}")
        return

    # YAMNet processes audio in frames of 0.96 seconds with a hop of 0.48 seconds.
    # Therefore, one set of scores is produced every 0.48 seconds.
    frame_hop_seconds = 0.48
    # Create an array of timestamps corresponding to the center of each frame's analysis window
    # The first score corresponds to analysis centered roughly at 0.48s (window 0 to 0.96s)
    # More accurately, the timestamp represents the beginning of the hop that leads to that score.
    times = np.arange(len(scores)) * frame_hop_seconds

    print(f"\n--- Detection Results ---")
    print(f"Audio Duration: {len(waveform)/sr:.2f} seconds")
    print(f"Number of YAMNet frames analyzed: {len(scores)}")

    found_event_count = 0
    for i, frame_scores in enumerate(scores):
        # Find the class with the highest score for this frame
        top_class_index = tf.argmax(frame_scores).numpy()
        top_class_name = class_names[top_class_index]
        top_class_score_logit = frame_scores[top_class_index].numpy()
        top_class_score_prob = tf.sigmoid(top_class_score_logit).numpy() # Convert logit to probability


        # Check if the top class is 'Cough' or related to throat sounds
        # You can adjust the keywords or add more specific class checks if needed.
        # Example YAMNet classes: "Cough", "Throat clearing", "Snoring"
        if "cough" in top_class_name.lower() or "throat" in top_class_name.lower():
            # The timestamp 'times[i]' marks the beginning of the 0.48s hop
            # for which this score was generated. The actual event could be within
            # the 0.96s window centered around (times[i] + 0.48s / 2) for a more precise center,
            # or simply stated to occur around this time.
            print(f"Frame {i}: Possible event '{top_class_name}' detected around {times[i]:.2f}s (Score: {top_class_score_prob:.2f})")
            found_event_count += 1

    if found_event_count == 0:
        print("No cough or throat clearing events detected as the top class in any frame.")
    else:
        print(f"\nFound {found_event_count} potential event frame(s).")

if __name__ == '__main__':
    # Replace "your_audio_file.wav" with the path to your audio file
    # Ensure the audio file is accessible by the script.
    detect_cough_in_audio(audio_file_path="input.wav")