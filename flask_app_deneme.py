from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import logging
from core import process_and_respond
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
from vllm import LLM
import logging
from llama_index.core import Settings
from configs import *

if __name__ == '__main__':
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'uploads'

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    @app.route('/')
    def home():
        logging.info("Rendering home page")
        return render_template('index.html')

    @app.route('/upload', methods=['POST'])
    def upload_file():
        logging.info("Received file upload request")
        
        if 'file' not in request.files:
            logging.error("No file part in the request")
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        if file.filename == '':
            logging.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            logging.info(f"File saved to {file_path}")
            
            question = request.form.get('question', '')
            logging.info(f"Processing file with question: {question}")
            
            response = process_and_respond(file_path, question)
            logging.info(f"Response generated: {response}")
            
            os.remove(file_path)  # Clean up the uploaded file after processing
            logging.info(f"File {file_path} removed")
            
            return jsonify({'response': response})

    logging.info("Starting Flask app")
    engine = LLM(model=MODEL_NAME, dtype="bfloat16", tensor_parallel_size=4)

    # Load model with vLLM
    llm = ChatHuggingFace(llm=engine)
    
    embedding = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cuda:1"},
        multi_process=True,
    )
    
    logging.info("Initializing LLM and embedding models")
    Settings.llm = llm
    Settings.embed_model = embedding
    app.run(debug=True)
