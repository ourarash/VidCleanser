from absl import app, flags
import os
from moviepy import VideoFileClip, concatenate_videoclips
from moviepy.Clip import Clip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.fx.FadeIn import FadeIn
from moviepy.video.fx.FadeOut import FadeOut

# === Define flags with defaults ===
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "clip_path",
    None,
    "Path to the input video file",
)
flags.DEFINE_string(
    "output_path", None, "Path for the output video file (default: <input>_cleaned.mov)"
)
flags.DEFINE_float("fade_duration", 0.3, "Fade in/out duration in seconds")
flags.DEFINE_float("silence_threshold", 0.03, "Silence threshold (linear amplitude)")
flags.DEFINE_float("chunk_duration", 0.5, "Chunk duration in seconds")
flags.DEFINE_bool("merge", True, "Merge consecutive loud segments")
flags.DEFINE_float("merge_gap_threshold", 0.05, "Max gap to merge segments in seconds")
flags.DEFINE_bool("fade_in_out", True, "Apply fade in/out effects")


# === Helper Functions ===
def load_video_clip(path):
    """Load a video file and handle errors gracefully."""
    try:
        return VideoFileClip(path)
    except Exception as e:
        print(f"Error loading video file: {e}")
        exit()


def detect_loud_segments(clip, chunk_duration, silence_threshold):
    """Detect loud segments in the video based on audio amplitude."""
    segments = []
    # Iterate over the video in chunks of chunk_duration seconds
    for t_start in range(0, int(clip.duration / chunk_duration)):
        start = t_start * chunk_duration
        end = min((t_start + 1) * chunk_duration, clip.duration)
        if end <= start:
            continue  # Skip invalid or zero-length chunks
        sub = clip.subclipped(start, end)  # Extract subclip for this chunk
        if sub.audio:
            # Convert audio to numpy array for analysis
            audio = sub.audio.to_soundarray(fps=16000)
            # Check if the chunk is loud enough
            if audio.size > 0 and audio.max() > silence_threshold:
                segments.append((start, end))  # Mark as a loud segment
    return segments


def merge_close_segments(segments, gap_threshold):
    """
    Merge consecutive segments if the gap between them is less than or equal to gap_threshold.

    Args:
        segments (list of tuple): List of (start, end) tuples representing segments.
        gap_threshold (float): Maximum allowed gap (in seconds) to merge segments.

    Returns:
        list of tuple: Merged list of (start, end) tuples.
    """
    if not segments:
        return []
    merged = []
    s, e = segments[0]
    for ns, ne in segments[1:]:
        # If the next segment starts within the threshold, extend the current segment
        if ns - e <= gap_threshold:
            e = ne
        else:
            merged.append((s, e))
            s, e = ns, ne
    merged.append((s, e))  # Add the last segment
    return merged


def crossfade_sequence(clips: list[Clip], overlap: float) -> CompositeVideoClip:
    """
    Build a composite video by sequencing clips with crossfade transitions.

    Each clip is placed sequentially, with optional fade-in and fade-out effects
    applied to create smooth transitions between clips.

    Args:
        clips (list[Clip]): List of video clips to sequence.
        overlap (float): Duration (in seconds) of the crossfade between clips.

    Returns:
        CompositeVideoClip: The resulting composite video with crossfades.
    """
    result = []
    t = 0  # Track the start time for each clip in the sequence
    for i, clip in enumerate(clips):
        c = clip.with_start(t)  # Set the start time for the current clip
        if i > 0:
            # Apply fade-in to all but the first clip
            c = c.with_effects([FadeIn(overlap)])
        if i < len(clips) - 1:
            # Apply fade-out to all but the last clip
            c = c.with_effects([FadeOut(overlap)])
        result.append(c)
        # Move the timeline forward, overlapping by the fade duration
        t += clip.duration - overlap
    return CompositeVideoClip(result)


# === Export ===
def export_final_video(final_video, output_path, subclips, clip, max_threads):
    """
    Export the final video to disk and handle cleanup.

    Args:
        final_video (VideoClip): The processed/composited video to export.
        output_path (str): Path to save the exported video file.
        subclips (list): List of subclip objects to close after export.
        clip (VideoClip): The original loaded video clip to close after export.
        max_threads (int): Number of threads to use for encoding.
    """
    try:
        print(f"Exporting to {output_path}...")
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            # ffmpeg_params=["-preset", "fast", "-crf", "20"],
            ffmpeg_params=["-preset", "slow", "-crf", "18", "-movflags", "+faststart"],
            threads=max_threads,
            bitrate="2500k",  # match source bitrate
        )
        print("Done exporting.")
    except Exception as e:
        print(f"Error during export: {e}")
    finally:
        # Clean up all resources to avoid file locks or memory leaks
        for c in subclips:
            c.close()
        final_video.close()
        clip.close()
        # Remove temp audio file if it exists
        if os.path.exists("temp-audio.m4a"):
            try:
                os.remove("temp-audio.m4a")
            except Exception as e:
                print(f"Warning: Could not remove temp-audio.m4a: {e}")


def main(argv):
    max_threads = os.cpu_count()
    print(f"Using {max_threads} threads for processing.")

    clip_path = FLAGS.clip_path
    output_path = FLAGS.output_path if FLAGS.output_path else clip_path + "_cleaned.mov"
    FADE_DURATION = FLAGS.fade_duration
    SILENCE_THRESHOLD = FLAGS.silence_threshold
    CHUNK_DURATION = FLAGS.chunk_duration
    MERGE_CONSECUTIVE_CLIPS = FLAGS.merge
    MERGE_GAP_THRESHOLD = FLAGS.merge_gap_threshold
    FADE_IN_OUT = FLAGS.fade_in_out

    clip = load_video_clip(clip_path)

    print("Detecting loud segments...")
    segments = detect_loud_segments(clip, CHUNK_DURATION, SILENCE_THRESHOLD)
    print(f"Detected {len(segments)} segments.")

    if MERGE_CONSECUTIVE_CLIPS:
        print("Merging close segments...")
        segments = merge_close_segments(segments, MERGE_GAP_THRESHOLD)

    if not segments:
        print("No loud segments found. Exiting.")
        clip.close()
        exit()

    print("Converting segments to subclips...")
    subclips = [clip.subclipped(start, end) for (start, end) in segments]
    print(f"Created {len(subclips)} subclips.")

    print("Creating final video...")
    if FADE_IN_OUT:
        final_video = crossfade_sequence(subclips, FADE_DURATION)
    else:
        final_video = concatenate_videoclips(subclips, method="compose")

    print("Exporting final video...")
    export_final_video(final_video, output_path, subclips, clip, max_threads)


if __name__ == "__main__":
    app.run(main)
