from absl import app, flags
from moviepy import VideoFileClip, concatenate_videoclips
import whisper
import json
import os
import torch
import re

# === Define flags ===
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "video_path",
    None,
    "Path to the input video file",
)
flags.DEFINE_string(
    "output_path", None, "Path for the output video file (default: <input>_cleaned.mov)"
)
flags.DEFINE_string(
    "transcript_path",
    None,
    "Path for the transcript JSON file (default: <video_path>transcript.json)",
)
flags.DEFINE_list(
    "fillers",
    ["um", "uh", "ah", "like", "you know", "i mean", "so"],
    "Comma-separated list of filler words to remove",
)


# === Helper to parse timestamp "00:01:02,480" -> seconds
def parse_timestamp(ts):
    h, m, s_ms = ts.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def main(argv):
    video_path = FLAGS.video_path
    output_path = (
        FLAGS.output_path if FLAGS.output_path else video_path + "_cleaned.mov"
    )
    transcript_path = FLAGS.transcript_path or (video_path + "transcript.json")
    fillers = set([f.strip().lower() for f in FLAGS.fillers])

    # === Transcribe with Whisper if transcript doesn't exist ===
    if not os.path.exists(transcript_path):
        print("Transcribing audio with Whisper...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        model = whisper.load_model("large", device=device)
        result = model.transcribe(video_path, word_timestamps=True)
        with open(transcript_path, "w") as f:
            json.dump(result, f)
        print("Transcript saved.")
    else:
        print("Transcript already exists. Skipping transcription.")

    # === Load transcript ===
    with open(transcript_path, "r") as f:
        data = json.load(f)

    words = data.get("transcription", [])

    # === Detect filler segments ===
    filler_segments = []
    for i, word_info in enumerate(words):
        word = re.sub(r"[^\w]+", "", word_info["text"].lower())

        # skip "so" at beginning of sentence (heuristic: skip if previous is "")
        if word == "so" and i > 0 and words[i - 1]["text"].strip() == "":
            continue

        if word in fillers:
            start_time = parse_timestamp(word_info["timestamps"]["from"])
            end_time = parse_timestamp(word_info["timestamps"]["to"])
            print(f"Filler: {word} @ {start_time:.2f} - {end_time:.2f}")
            filler_segments.append((start_time, end_time))

    print(f"\nDetected {len(filler_segments)} filler segments.")
    if not filler_segments:
        print("No filler words found. Exiting.")
        return

    # === Remove filler segments from video ===
    clip = VideoFileClip(video_path)
    parts = []
    prev_end = 0

    for start, end in filler_segments:
        if start > prev_end:
            parts.append(clip.subclipped(prev_end, start))
        prev_end = max(prev_end, end)

    if prev_end < clip.duration:
        parts.append(clip.subclipped(prev_end, clip.duration))

    # === Export cleaned video ===
    final = concatenate_videoclips(parts, method="compose")
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        ffmpeg_params=["-crf", "18", "-preset", "fast"],
    )


if __name__ == "__main__":
    app.run(main)
