import json
import unittest
from io import BytesIO
from unittest.mock import patch
from main import app


def mock_generate_content(chunk):
    mock_response = {
        "entities": [
            {
                "entity": "COVID-19",
                "context": "... was observed in patients with COVID-19",
                "start": 30,
                "end": 45,
            },
            {
                "entity": "ERK1",
                "context": "... elevated levels of ERK1 were seen",
                "start": 10,
                "end": 15,
            },
        ]
    }
    return type("MockResponse", (object,), {"text": json.dumps(mock_response)})


# Mock PDF text extraction
def mock_extract_text_from_pdf(pdf_path):
    return "This is a sample PDF content mentioning COVID-19 and ERK1 in biological context."


class TestNERAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    @patch("main.model.generate_content", side_effect=mock_generate_content)
    @patch("main.extract_text_from_pdf", side_effect=mock_extract_text_from_pdf)
    def test_upload_pdf(self, mock_generate, mock_extract_pdf):
        # Create a mock PDF file
        data = {
            "file": (BytesIO(b"%PDF-1.4\n%Valid PDF content"), "test.pdf"),
        }

        response = self.app.post(
            "/api/v1/extract", content_type="multipart/form-data", data=data
        )
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertIn("entities", json_data)

        self.assertEqual(json_data["entities"][0]["entity"], "COVID-19")
        self.assertEqual(json_data["entities"][1]["entity"], "ERK1")

    # Test case when no file is uploaded
    def test_no_file(self):
        response = self.app.post("/api/v1/extract", content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertIn("error", json_data)
        self.assertEqual(json_data["error"], "No file part")

    # Test case when an invalid file format is uploaded (not a PDF)
    def test_invalid_file_format(self):
        data = {
            "file": (BytesIO(b"This is not a PDF"), "test.txt"),
        }

        response = self.app.post(
            "/api/v1/extract", content_type="multipart/form-data", data=data
        )
        self.assertEqual(response.status_code, 415)

        json_data = response.get_json()
        self.assertIn("error", json_data)
        self.assertEqual(
            json_data["error"], "File format not supported, please upload a PDF"
        )


if __name__ == "__main__":
    unittest.main()
