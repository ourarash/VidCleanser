import unittest
from unittest.mock import MagicMock, patch
from filler_remover import parse_timestamp, detect_filler_segments, remove_filler_segments_from_video

class TestFillerRemover(unittest.TestCase):
    def test_parse_timestamp(self):
        self.assertAlmostEqual(parse_timestamp("00:01:02,480"), 62.48)
        self.assertAlmostEqual(parse_timestamp("01:00:00,000"), 3600.0)

    def test_detect_filler_segments(self):
        words = [
            {"text": "Um", "timestamps": {"from": "00:00:01,000", "to": "00:00:01,500"}},
            {"text": "hello", "timestamps": {"from": "00:00:01,500", "to": "00:00:02,000"}},
            {"text": "uh", "timestamps": {"from": "00:00:02,000", "to": "00:00:02,300"}},
        ]
        fillers = {"um", "uh"}
        segments = detect_filler_segments(words, fillers)
        self.assertEqual(segments, [(1.0, 1.5), (2.0, 2.3)])

    @patch("filler_remover.VideoFileClip")
    def test_remove_filler_segments_from_video(self, MockVideoFileClip):
        mock_clip = MagicMock()
        mock_clip.duration = 10
        MockVideoFileClip.return_value = mock_clip

        # Patch the correct method: subclipped, not subclip
        mock_clip.subclipped.side_effect = lambda start, end: (start, end)

        segments = [(1, 2), (4, 5)]
        parts = remove_filler_segments_from_video("fake.mp4", segments)
        # Should return [(0,1), (2,4), (5,10)]
        self.assertEqual(parts, [(0,1), (2,4), (5,10)])

if __name__ == "__main__":
    unittest.main()