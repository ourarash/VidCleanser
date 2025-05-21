# VidCleaner

A Python toolkit for automatically removing silent sections and filler words from video files using [moviepy](https://zulko.github.io/moviepy/) and [OpenAI Whisper](https://github.com/openai/whisper).  
It analyzes the audio track, detects loud segments or filler words, and outputs a new video with only the "meaningful" parts, optionally applying crossfades.

---

## Features

- **Silence Remover:** Detects and removes silent sections from video/audio files
- **Filler Remover:** Detects and removes filler words (e.g., "um", "uh", "like") using Whisper transcription
- Adjustable silence threshold and chunk duration
- Optionally merges close loud segments
- Optional fade-in/out (crossfade) between segments
- Command-line interface using [absl-py](https://github.com/abseil/abseil-py)
- Easy integration with Whisper for transcript-based editing

---

## Installation

```bash
/opt/homebrew/bin/python3.11 -m venv venv311
source venv311/bin/activate
pip install --upgrade pip
pip install moviepy absl-py numpy torch openai-whisper
```

---

## Silence Remover Usage

```bash
python silence_remover.py \
  --clip_path=/path/to/input.mov \
  --output_path=/path/to/output.mov \
  --fade_duration=0.3 \
  --silence_threshold=0.03 \
  --chunk_duration=0.5 \
  --merge=True \
  --merge_gap_threshold=0.05 \
  --fade_in_out=True
```

**Arguments:**

- `--clip_path` (required): Path to the input video file
- `--output_path`: Path for the output video file (default: `<input>_cleaned.mov`)
- `--fade_duration`: Fade in/out duration in seconds (default: 0.3)
- `--silence_threshold`: Silence threshold (linear amplitude, default: 0.03)
- `--chunk_duration`: Chunk duration in seconds (default: 0.5)
- `--merge`: Merge consecutive loud segments (default: True)
- `--merge_gap_threshold`: Max gap to merge segments in seconds (default: 0.05)
- `--fade_in_out`: Apply fade in/out effects (default: True)
- `--padding`: Padding duration (in seconds) to add before and after each loud segment (default: 0.2)

---

## Filler Remover Usage

The filler remover uses Whisper to transcribe your video and then removes segments containing filler words.

```bash
python filler_remover.py \
  --video_path=/path/to/input.mov \
  --output_path=/path/to/output.mov \
  --fillers=um,uh,ah,like,you know,i mean,so
```

**Arguments:**

- `--video_path` (required): Path to the input video file
- `--output_path`: Path for the output video file (default: `<input>_cleaned.mov`)
- `--transcript_path`: Path for the transcript JSON file (default: `<video_path>transcript.json`)
- `--fillers`: Comma-separated list of filler words to remove (default: `um,uh,ah,like,you know,i mean,so`)

**How it works:**
1. Transcribes the video using Whisper (if a transcript does not already exist).
2. Detects filler words and their timestamps.
3. Removes those segments from the video.
4. Exports a cleaned video without filler words.

---

## Example

```bash
python silence_remover.py --clip_path=example.mov
python filler_remover.py --video_path=example.mov
```

---

## Testing

Run unit tests with:

```bash
python -m unittest discover tests
```

---

## License

MIT License
