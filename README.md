# Chat2Mistral

This project implements a Retrieval-Augmented Generation (RAG) application utilizing a hybrid search mechanism, combining keyword and vector search for document retrieval. It uses the LlamaIndex framework and integrates language models and embedding models from LangChain and Hugging Face. Additionally, a Flask app is provided for running the application.

## Features

- **Hybrid Search:** Combines BM25 keyword search and vector search to retrieve the most relevant documents.
- **Language Models:** Utilizes models from LangChain and Hugging Face for generating responses.
- **Flask Integration:** Provides a Flask app for easy deployment and usage.
- **Markdown Responses:** Ensures responses are formatted in markdown.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/g-hano/Chat2Mistral.git
cd Chat2Mistral
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the flask app
```bash
python flask_app.py
```

## Configuration

The application configuration can be adjusted in the `configs` section of the code. Key configuration parameters include:

- `MODEL_NAME`: The name of the language model (e.g., `mistralai/Mistral-7B-Instruct-v0.3`).
- `EMBEDDING_NAME`: The name of the embedding model (e.g., `nomic-ai/nomic-embed-text-v1.5`).
- `TEMPERATURE`: The temperature setting for the language model.
- `CONTEXT_WINDOW`: The context window size for the model.
- `TOP_K`: The number of top results to retrieve.
- `CHUNK_SIZE`: The chunk size for document splitting.
- `CHUNK_OVERLAP`: The overlap size between chunks.
- `SYSTEM_PROMPT`: The system prompt provided to the language model.
- `DEVICE`: The device to run the models on, default to (`auto`) for handling multi-gpus.

### ChatEngine Class

The `ChatEngine` class is responsible for handling the chat interactions with the language model.

```python
class ChatEngine:
    def __init__(self, retriever):
        self.retriever = retriever
        self.params = SamplingParams(...)

    def ask_question(self, question, llm):
        question = "[INST]" + question + "[/INST]"
        results = self.retriever.best_docs(question) # get most relevant docs
        document = [doc.text for doc, sc in results]
        chat_history = SYSTEM_PROMPT + "\n\n" + f"Question: {question}\n\nDocument: {document}"
        response = llm.generate(chat_history, self.params) # ask llm
        return response[0].outputs[0].text
```

### HybridRetriever Class

The `HybridRetriever` class combines BM25 and vector search methods to retrieve relevant documents.

```python
class HybridRetriever:
    def __init__(self, bm25_retriever: BM25Retriever, vector_retriever: VectorIndexRetriever):
        self.top_k = vector_retriever._similarity_top_k + bm25_retriever._similarity_top_k

    def retrieve(self, query: str):
        query = "[INST] " + " [/INST]"
        bm25_results = self.bm25_retriever.retrieve(query)
        vector_results = self.vector_retriever.retrieve(query)

        combined_results = {}
        for result in bm25_results:
            combined_results[result.node.text] = {'score': result.score}

        for result in vector_results:
            if result.node.text in combined_results:
                combined_results[result.node.text]['score'] += result.score
            else:
                combined_results[result.node.text] = {'score': result.score}

        combined_results_list = sorted(combined_results.items(), key=lambda item: item[1]['score'], reverse=True)
        return combined_results_list

    def best_docs(self, query: str):
        top_results = self.retrieve(query)
        return [(Document(text=text), score) for text, score in top_results]
```

---
All code belongs to [Cihan Yalçın](https://www.linkedin.com/in/chanyalcin/)
Huge thanks to [Josh Longenecker](https://www.linkedin.com/in/josh-longenecker-5a6902226/) for his assistance with multi-GPU configurations and example [notebook](https://github.com/jlonge4/gen_ai_utils/blob/main/vllm_vs_deepspeed.ipynb).
