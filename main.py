import os
from flask import Flask, request, jsonify
import pdfplumber
from gliner import GLiNER

app = Flask(__name__)

model_name = "urchade/gliner_large_bio-v0.1"
gliner_model = GLiNER.from_pretrained(model_name)

# TODO: Come back to labels
labels = ["B-Disease", "I-Disease", "B-Chemical", "I-Chemical", "B-Protein", "I-Protein"] 

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
    chunks = split_text_into_chunks(text)
    entities = []
    
    for chunk in chunks:
        try:
            ner_results = gliner_model.predict_entities(chunk, labels=labels)

            for entity in ner_results:
                # TODO Start and end token not text
                surrounding_text = text[(entity["start"] - 100): entity["end"] + 100]
                entities.append({
                    "entity": entity["text"],
                    "label": entity["label"],
                    "context": f"... {surrounding_text} ...",
                    "start": entity["start"],
                    "end": entity["end"],
                })
        except Exception as e:
            print("Error during NER processing:", e)
    return entities


@app.route('/api/v1/extract', methods=['POST'])
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
        return jsonify({"error": "File format not supported, please upload a PDF"}), 415

if __name__ == '__main__':
    app.run(debug=True)
