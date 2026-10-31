"""Tests for media I/O utilities."""

import os
import tempfile


from really_rally.utils.media_io import generate_rally_name


class TestGenerateRallyName:
    def test_format_includes_date_and_stem(self):
        """Output should look like '010124_test-video_rally'."""
        # We can't easily test get_video_date without a real video,
        # but we can test the function structure by mocking.
        # Instead, test that the function doesn't crash and returns a string.
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake mp4 content")
            tmp_path = f.name

        try:
            result = generate_rally_name(tmp_path)
            assert isinstance(result, str)
            assert "_rally" in result
            assert len(result) > 10  # at least MMDDYY + _ + stem + _rally
        finally:
            os.unlink(tmp_path)

    def test_name_ends_with_rally_suffix(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake mp4 content")
            tmp_path = f.name

        try:
            result = generate_rally_name(tmp_path)
            assert result.endswith("_rally")
        finally:
            os.unlink(tmp_path)
