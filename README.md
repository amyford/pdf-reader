# PDF reader - NER

This project implements a Named Entity Recognition (NER) API that extracts biological entities from PDFs. It uses **Vertex AI** for processing the text and identifying biological entities. The API accepts PDF files as input, extracts the text from them, and then sends the text to Vertex AI's `Gemini` model to perform NER.


## Requirements
- Python 3.8+
- Vertex AI account and credentials

## Setup

1. Clone the repository:

   ```
   git clone https://github.com/amyford/pdf-reader.git
   cd pdf-reader
   ```


2. Install dependencies:

    ```
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. Set up your Google Cloud project for Vertex AI and authenticate
    ```
    gcloud auth application-default login
    export GOOGLE_CLOUD_PROJECT=your-project-id
    ```

For more details, please See the GCloud website.
Configure your API to access Vertex AI API.


## Running locally

```
python3 main.py
```


## Running in the Cloud
Deploy your service to Cloud Run

```
gcloud run deploy --source .
```
Then use the url provided.

# Usage

Example:
```
curl -X POST -F "file=@path/to/file.pdf" http://127.0.0.1:5000/api/v1/extract
```

Example Response:
```
{
  "entities": [
    {
      "entity": "COVID-19",
      "context": "... was observed in patients with COVID-19",
      "start": 30,
      "end": 45
    },
    {
      "entity": "ERK1",
      "context": "... elevated levels of ERK1 were seen",
      "start": 10,
      "end": 15
    }
  ]
}
```


# Testing
```
python -m unittest test_main.py
```
