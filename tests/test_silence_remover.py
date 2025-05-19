import unittest
from unittest.mock import MagicMock, patch
import numpy as np  # <-- Add this import
from silence_remover import detect_loud_segments, merge_close_segments


class TestSilenceRemover(unittest.TestCase):
    def test_merge_close_segments(self):
        segments = [(0, 1), (1.01, 2), (3, 4)]

        merged = merge_close_segments(segments, gap_threshold=0.05)

        self.assertEqual(merged, [(0, 2), (3, 4)])

    def test_merge_close_segments_empty(self):
        self.assertEqual(merge_close_segments([], 0.1), [])

    @patch("silence_remover.VideoFileClip")
    def test_detect_loud_segments(self, MockVideoFileClip):
        mock_clip = MagicMock()
        mock_clip.duration = 3

        # First chunk: loud, Second chunk: silent, Third chunk: loud
        mock_audio_loud = MagicMock()
        mock_audio_loud.to_soundarray.return_value = np.array([0.02, 0.04])
        mock_subclip_loud = MagicMock()
        mock_subclip_loud.audio = mock_audio_loud

        mock_audio_silent = MagicMock()
        mock_audio_silent.to_soundarray.return_value = np.array([0.01, 0.02])
        mock_subclip_silent = MagicMock()
        mock_subclip_silent.audio = mock_audio_silent

        mock_audio_loud2 = MagicMock()
        mock_audio_loud2.to_soundarray.return_value = np.array([0.06, 0.05])
        mock_subclip_loud2 = MagicMock()
        mock_subclip_loud2.audio = mock_audio_loud2

        mock_clip.subclipped.side_effect = [
            mock_subclip_loud,
            mock_subclip_silent,
            mock_subclip_loud2,
        ]

        segments = detect_loud_segments(
            mock_clip, chunk_duration=1, silence_threshold=0.03
        )

        self.assertEqual(segments, [(0, 1), (2, 3)])

    @patch("silence_remover.VideoFileClip")
    def test_detect_loud_segments_silent(self, MockVideoFileClip):
        mock_clip = MagicMock()
        mock_clip.duration = 2
        mock_audio = MagicMock()
        mock_audio.to_soundarray.return_value = np.array([0.01, 0.02])
        mock_subclip = MagicMock()
        mock_subclip.audio = mock_audio
        mock_clip.subclipped.return_value = mock_subclip

        segments = detect_loud_segments(
            mock_clip, chunk_duration=1, silence_threshold=0.03
        )
        self.assertEqual(segments, [])


if __name__ == "__main__":
    unittest.main()
