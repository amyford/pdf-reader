import unittest
from unittest.mock import patch
import io

from main import app

class TestNERApi(unittest.TestCase):

    # Setup the test client and mock paths
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    # Test the case where no file is uploaded
    def test_no_file(self):
        response = self.app.post('/api/v1/extract')
        self.assertEqual(response.status_code, 400)
        self.assertIn("No file part", response.json["error"])

    # Test the case where an empty file name is provided
    def test_empty_filename(self):
        data = {
            'file': (io.BytesIO(b"Test PDF content"), '')
        }
        response = self.app.post('/api/v1/extract', content_type='multipart/form-data', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn("No selected file", response.json["error"])

    # Test when a non-PDF file is uploaded
    def test_non_pdf_file(self):
        data = {
            'file': (io.BytesIO(b"Not a PDF"), 'test.txt')
        }
        response = self.app.post('/api/v1/extract', content_type='multipart/form-data', data=data)
        self.assertEqual(response.status_code, 415)
        self.assertIn("File format not supported", response.json["error"])

    # Test with a valid PDF file and mock PDF text extraction and NER processing
    @patch('main.extract_text_from_pdf')
    @patch('main.perform_ner')
    def test_valid_pdf(self, mock_perform_ner, mock_extract_text_from_pdf):
        # Simulate extracted text from PDF
        mock_extract_text_from_pdf.return_value = "This is a sample text from PDF."

        # Simulate NER results
        mock_perform_ner.return_value = [
            {"entity": "Protein", "label": "B-Protein", "context": "... Protein ABC ...", "start": 10, "end": 20}
        ]

        # Create a fake PDF upload
        data = {
            'file': (io.BytesIO(b"%PDF-1.4 Test PDF content"), 'test.pdf')
        }

        # Make the POST request
        response = self.app.post('/api/v1/extract', content_type='multipart/form-data', data=data)

        # Assertions
        self.assertEqual(response.status_code, 200)
        self.assertIn("entities", response.json)
        self.assertEqual(len(response.json["entities"]), 1)
        self.assertEqual(response.json["entities"][0]["entity"], "Protein")

        mock_extract_text_from_pdf.assert_called_once()
        mock_perform_ner.assert_called_once_with("This is a sample text from PDF.")

if __name__ == '__main__':
    unittest.main()
