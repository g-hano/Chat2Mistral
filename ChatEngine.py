from llama_index.core.base.llms.types import ChatMessage, MessageRole
import logging
logging.basicConfig(level=logging.INFO)

class ChatEngine:
    def __init__(self, retriever):
        """
        Initializes the ChatEngine with a retriever and a language model.

        Args:
            retriever (HybridRetriever): An instance of a retriever to fetch relevant documents.
            model_name (str): The name of the language model to be used.
            context_window (int, optional): The maximum context window size for the language model. Defaults to 32000.
            temperature (float, optional): The temperature setting for the language model. Defaults to 0.
        """

        self.retriever = retriever
        self.chat_history = []

    def ask_question(self, question, llm):
        """
        Asks a question to the language model, using the retriever to fetch relevant documents.

        Args:
            question (str): The question to be asked.

        Returns:
            str: The response from the language model in markdown format.
        """
        
        question = "[INST]" + question + "[/INST]"
        
        results = self.retriever.best_docs(question)
        document = [doc.text for doc, sc in results]
        logging.info(f"Created Document - len docs:{len(document)}")
        self.chat_history.append(f"Question: {question}")
        
        self.chat_history.append(f"\nDocument: {document}")
        logging.info("Created Chat History")
        logging.info("Asking LLM")
        #response = llm.chat(self.chat_history)
        response = llm.generate(self.chat_history)
        logging.info("Got Response from LLM, Returning")
        return response.message.content
