import os
from flask import Flask, request, jsonify
import pdfplumber
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Initialize the Flask app
app = Flask(__name__)

# Load BioBERT model and tokenizer
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Function to extract text from a PDF using pdfplumber
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() or ''  # Handle case where text extraction might return None
    return text

# Function to split the text into chunks using tokenizer's max length
def split_text_into_chunks(text, tokenizer, max_length=512):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    input_ids = tokens['input_ids'].squeeze(0).tolist()  # Convert tensor to list of token ids
    
    # Split tokens into chunks of max_length
    chunks = []
    for i in range(0, len(input_ids), max_length - 2):  # Account for [CLS] and [SEP] tokens
        chunk = input_ids[i:i + max_length - 2]
        # Add [CLS] and [SEP] tokens
        chunk = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
        chunks.append(chunk)
    
    return chunks

# Function to merge subword tokens into complete entities
def merge_subword_tokens(ner_results):
    merged_entities = []
    previous_entity = None
    for entity in ner_results:
        if entity['entity_group'] == previous_entity:
            # Merge with the previous entity
            merged_entities[-1]['word'] += " " + entity['word']
            merged_entities[-1]['score'] = max(merged_entities[-1]['score'], entity['score'])
        else:
            # Start a new entity
            merged_entities.append(entity)
        previous_entity = entity['entity_group']
    return merged_entities

# Function to perform NER using BioBERT on tokenized text chunks
def perform_ner(text):
    # Tokenize and split text into chunks that fit the model's token limit
    chunks = split_text_into_chunks(text, tokenizer)

    entities = []
    for chunk in chunks:
        # Decode the chunk back to string for NER processing
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)

        try:
            ner_results = ner_pipeline(chunk_text)
            merged_entities = merge_subword_tokens(ner_results)

            for entity in merged_entities:
                entities.append({
                    "entity": entity["entity_group"],
                    "word": entity["word"],
                    "score": float(entity["score"])  # Convert to Python float
                })
        except Exception as e:
            print("Error during NER processing:", e)
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

        # Perform NER on the extracted text using BioBERT
        entities = perform_ner(extracted_text)

        # Clean up the temp file
        os.remove(pdf_path)

        return jsonify({"entities": entities}), 200
    else:
        return jsonify({"error": "File format not supported, please upload a PDF"}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
