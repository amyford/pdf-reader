import vertexai
from vertexai.generative_models import GenerativeModel
import os
from flask import Flask, request, jsonify
import pdfplumber
import json
from time import sleep
app = Flask(__name__)

# NB: if running locally may need to add an api_key
vertexai.init(project="pdf-reader-438611", location="europe-west1")

instructions = """
        Extract the biological entities from the text below.

        Return the list of entities.

        The output must be in JSON.

        [{
            "entity",
            "context",
            "start",
            "end",
        }]

        Where:
            "entity": Name of the identified medical entity,
            "context": Text surrounding the entity for clarity. Make sure to keep whole words. This should be as it is in the text"
            "start": The start position of the entity in the context with respect to the original text,
            "end": The end position of the entity in the context with respect to the original text,

        For example
        {"entities": [
            {
                "entity": "COVID-19",
                "context": "... was observed in patients with COVID-19",
                "start": 119,
                "end": 140
            },
            {
                "entity": "ERK1",
                "context": "... elevated levels of ERK1 were seen in patients with COVID-19",
                "start": 119,
                "end": 140
            }
        ]}

"""

model = GenerativeModel(
    model_name="gemini-1.5-flash-002", 

    generation_config={"response_mime_type": "application/json"},
    system_instruction=instructions
)

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() or ''  # Handle case where text extraction might return None
    return text


def split_text_into_chunks(text, max_length=512):
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    return chunks

def perform_ner(text):
    entities =[]
    chunks = split_text_into_chunks(text)

    for chunk in chunks[0:5]:
        try: 
            response = model.generate_content(chunk)
            entities_in_chunk = json.loads(response.text)
            print(entities_in_chunk)
            entities = [*entities, *entities_in_chunk["entities"]]

            print(entities)
        except Exception as e:
            print("Error extracting entities", e)

        # Hit Gemini limit on free plan! - sleep between requests
        sleep(1)
    
    return entities

@app.route('/api/v1/extract', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.pdf'):
        pdf_path = os.path.join("/tmp", file.filename)
        file.save(pdf_path)

        extracted_text = extract_text_from_pdf(pdf_path)
        entities = perform_ner(extracted_text)

        # Clean up the temp file
        os.remove(pdf_path)

        return jsonify({"entities": entities}), 200
    else:
        return jsonify({"error": "File format not supported, please upload a PDF"}), 415

if __name__ == '__main__':
    app.run(debug=True)
