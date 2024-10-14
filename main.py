import os
from flask import Flask, request, jsonify
import pdfplumber
import spacy

# Initialize the Flask app
app = Flask(__name__)

# Load the spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Function to extract text from a PDF using pdfplumber
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to perform NER on the extracted text
def perform_ner(text):
    doc = nlp(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return entities

# Route to upload a PDF and perform NER
@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.pdf'):
        # Save the uploaded PDF temporarily
        pdf_path = os.path.join("/tmp", file.filename)
        file.save(pdf_path)

        # Extract text from the PDF
        extracted_text = extract_text_from_pdf(pdf_path)

        print("pdf_path", pdf_path)

        

        # # Perform NER on the extracted text
        # entities = perform_ner(extracted_text)

        # Clean up the temp file
        os.remove(pdf_path)

        return jsonify({"entities": "AAA"}), 200
        # return jsonify({"entities": entities}), 200
    else:
        return jsonify({"error": "File format not supported, please upload a PDF"}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)