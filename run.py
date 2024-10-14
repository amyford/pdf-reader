import os
from flask import Flask, request, jsonify
import pdfplumber
from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline

app = Flask(__name__)

model_name = "allenai/scibert_scivocab_uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name) 

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)


def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to split the text into chunks using tokenizer's max length
def split_text_into_chunks(text, tokenizer, max_length=512):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    input_ids = tokens['input_ids'].squeeze(0).tolist()  # Convert tensor to list of token ids
    
    chunks = []
    for i in range(0, len(input_ids), max_length - 2):  # [CLS] and [SEP] tokens
        chunk = input_ids[i:i + max_length - 2]
        chunk = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
        chunks.append(chunk)
    
    return chunks
    
def merge_subword_tokens(ner_results):
    merged_entities = []
    current_entity = ""
    for token in ner_results:
        word = token["word"]
        # If it's a continuation token (starts with ##), append to current entity
        if word.startswith("##"):
            current_entity += word[2:]  # Remove the ##
        else:
            if current_entity:  # If there's an existing entity, append it
                merged_entities.append(current_entity)
            current_entity = word  # Start new entity
        
    # Append the last entity
    if current_entity:
        merged_entities.append(current_entity)

    return merged_entities


# Function to perform NER using SciBERT on tokenized text chunks
def perform_ner(text):
    # Tokenize and split text into chunks that fit the model's token limit
    chunks = split_text_into_chunks(text, tokenizer)

    entities = []
    for chunk in chunks:
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)

        try:
            ner_results = ner_pipeline(chunk_text)
            merged_entities = merge_subword_tokens(ner_results)
        except Exception as e:
            print("Error", e)
        for entity in merged_entities:
            # TODO - what to return
            entities.append(entity)
    return entities


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

        extracted_text = extract_text_from_pdf(pdf_path)
        entities = perform_ner(extracted_text)

        # Clean up the temp file
        os.remove(pdf_path)

        return jsonify({"entities": entities}), 200
    else:
        return jsonify({"error": "File format not supported, please upload a PDF"}), 400

if __name__ == '__main__':
    app.run(debug=True)
