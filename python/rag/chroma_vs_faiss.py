import argparse
import os
import glob
import gradio as gr
import numpy as np
import plotly.graph_objects as go
import signal
import sys
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_core.callbacks import StdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from sklearn.manifold import TSNE

class Rag:
    def __init__(self, debug_mode):
        self._debug_mode = debug_mode
        self._model = "gpt-4o-mini"
        self._db_name = "vector_db"
        self._chunks = self._generate_document_chunks()
        self._selected_vector_store = None
        self._selected_embeddings = None
        # We need a stateful conversation_retrieval_chain, so we make it a member variable
        self._conversation_retrieval_chain = None
        

    def run(self):
        with gr.Blocks() as ui:
            with gr.Row():
                chatbot_output = gr.Chatbot(height=500, type="messages")
            with gr.Row():
                selected_vector_store = gr.Dropdown(["Chroma", "FAISS"], label="Select vector store", value="Chroma")
            with gr.Row():
                selected_embeddings = gr.Dropdown(["OpenAIEmbeddings", "HuggingFaceEmbeddings"],
                                                  label="Select Embeddings", value="OpenAIEmbeddings")
            with gr.Row():
                message_box = gr.Textbox(label="Chat with our AI Assistant:")
            with gr.Row():
                clear_button = gr.Button("Clear")

            message_box.submit(self._chat, inputs=[message_box, chatbot_output, selected_vector_store, selected_embeddings],
                               outputs=[message_box, chatbot_output])

            # Clear out chatbot_output and message_box
            selected_vector_store.change(lambda: [None, ""], inputs=None, outputs=[chatbot_output, message_box], queue=False)
            # Clear out chatbot_output and message_box
            selected_embeddings.change(lambda: [None, ""], inputs=None, outputs=[chatbot_output, message_box], queue=False)
            # Clear out chatbot_output and message_box
            clear_button.click(lambda: [None, ""], inputs=None, outputs=[chatbot_output, message_box], queue=False)

        ui.launch(inbrowser=True)

    def _chat(self, message, history, selected_vector_store, selected_embeddings):
        # First add user message to history
        history += [{"role": "user", "content": message}]
        # If it's a first time or any changes to vectorstore/embeddings, we need to re-initiate conversation_retrieval_chain
        if (self._selected_vector_store is None or self._selected_vector_store != selected_vector_store
            or self._selected_embeddings is None or self._selected_embeddings != selected_embeddings):
            self._selected_vector_store = selected_vector_store
            self._selected_embeddings = selected_embeddings
            # Set up embeddings
            embeddings = None
            if selected_embeddings == 'OpenAIEmbeddings':
                embeddings = OpenAIEmbeddings()
            else:
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            # Set up vectorstore
            vectorstore = None
            if selected_vector_store == 'Chroma':
                vectorstore = self._setup_chroma_vector_store(embeddings)
            else:
                vectorstore = self._setup_faiss_vector_store(embeddings)
            # create a new Chat with OpenAI
            llm = ChatOpenAI(temperature=0.7, model_name=self._model)
            # set up the conversation memory for the chat
            memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
            # the retriever is an abstraction over the VectorStore that will be used during RAG
            # k is how many chunks to use
            retriever = vectorstore.as_retriever(search_kwargs={"k": 25})
            # putting it together: set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
            self._conversation_retrieval_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                                             retriever=retriever,
                                                                             memory=memory,
                                                                             callbacks=[StdOutCallbackHandler()] if self._debug_mode else None)

        if self._debug_mode:
            retrieved_docs = self._conversation_retrieval_chain.retriever.get_relevant_documents(message)
            print(f'\nRetrieved docs from {selected_vector_store} vector store are: ', retrieved_docs)


        result = self._conversation_retrieval_chain.invoke({"question": message})
        answer = result["answer"]
        if self._debug_mode:
            print("\nAnswer: ", answer)
        history += [{"role": "assistant", "content": answer}]
        # 1st return is to clear out message_box. 2nd return is to show output to chatbot_box
        return "", history

    def _generate_document_chunks(self):
        # Fetch all the folders under knowledge-base/*
        folders = glob.glob("knowledge-base/*")
        text_loader_kwargs = {'encoding': 'utf-8'}
        documents = []
        for folder in folders:
            # only get the folder name without paths and treat it as doc_type like products, employees, etc.
            doc_type = os.path.basename(folder)
            loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
            folder_docs = loader.load()
            for doc in folder_docs:
                doc.metadata["doc_type"] = doc_type
                documents.append(doc)
        # Set each chunk to be 1000 characters and each chunk overlap to be 200.
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        if (self._debug_mode):
            print(f'There are total {len(chunks)} chunks')
            doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
            print(f"Document types found: {', '.join(doc_types)}")
        return chunks

    def _setup_chroma_vector_store(self, embeddings):
        # Delete any existing chroma vector db
        if os.path.exists(self._db_name):
            Chroma(persist_directory=self._db_name, embedding_function=embeddings).delete_collection()

        # Create chroma vectorstore
        vectorstore = Chroma.from_documents(documents=self._chunks, embedding=embeddings, persist_directory=self._db_name)
        if self._debug_mode:
            print(f"Chroma Vectorstore created with {vectorstore._collection.count()} documents")
            collection = vectorstore._collection
            sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
            dimensions = len(sample_embedding)
            print(f"One Chroma vector has {dimensions:,} dimensions")
            self._visualize_chroma_vector_store_in_3d(collection)
        return vectorstore

    def _setup_faiss_vector_store(self, embeddings):
        vectorstore = FAISS.from_documents(self._chunks, embedding=embeddings)
        if self._debug_mode:
            total_vectors = vectorstore.index.ntotal
            dimensions = vectorstore.index.d
            print(f"\nFaiss vector store: There are {total_vectors} vectors with {dimensions:,} dimensions in the vector store")
            self._visualize_faiss_vector_store_in_3d(vectorstore)
        return vectorstore

    def _visualize_faiss_vector_store_in_3d(self, vectorstore):
        # Prework
        vectors = []
        documents = []
        doc_types = []
        colors = []
        color_map = {'products':'blue', 'employees':'green', 'contracts':'red', 'company':'orange'}

        total_vectors = vectorstore.index.ntotal

        for i in range(total_vectors):
            vectors.append(vectorstore.index.reconstruct(i))
            doc_id = vectorstore.index_to_docstore_id[i]
            document = vectorstore.docstore.search(doc_id)
            documents.append(document.page_content)
            doc_type = document.metadata['doc_type']
            doc_types.append(doc_type)
            colors.append(color_map[doc_type])
            
        vectors = np.array(vectors)

        self._plot_vector_store_in_3d(vectors, documents, doc_types, colors, 'FAISS')

    def _visualize_chroma_vector_store_in_3d(self, collection):
        # Prework
        result = collection.get(include=['embeddings', 'documents', 'metadatas'])
        vectors = np.array(result['embeddings'])
        documents = result['documents']
        doc_types = [metadata['doc_type'] for metadata in result['metadatas']]
        # apply different colors to each doc_type
        colors = [['blue', 'green', 'red', 'orange'][['products', 'employees', 'contracts', 'company'].index(t)] for t in doc_types]
            
        tsne = TSNE(n_components=3, random_state=42)
        reduced_vectors = tsne.fit_transform(vectors)

        self._plot_vector_store_in_3d(vectors, documents, doc_types, colors, 'Chroma')
        
    def _plot_vector_store_in_3d(self, vectors, documents, doc_types, colors, vector_store_name):
        tsne = TSNE(n_components=3, random_state=42)
        reduced_vectors = tsne.fit_transform(vectors)

        # Create the 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            z=reduced_vectors[:, 2],
            mode='markers',
            marker=dict(size=5, color=colors, opacity=0.8),
            text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
            hoverinfo='text'
        )])

        fig.update_layout(
            title=f'3D {vector_store_name} Vector Store Visualization',
            scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
            width=900,
            height=700,
            margin=dict(r=20, b=10, l=10, t=40)
        )

        # This will open a new tab in a browser and show the 3d graph
        fig.show()
        

# Function to handle closing the interface
# The 2nd parameter is frame which is required by the signal handler signature
# Since it's not used, use _ instead
def handle_exit_signal(signal, _):
    print("Exiting gradio!")
    gr.close_all()
    sys.exit(0)  # Exit the program

def str2bool(value):
    """Convert string input to boolean."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "yes", "y", "1"):
        return True
    elif value.lower() in ("false", "f", "no", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected: 'true' or 'false'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This script with required --debug-mode argument")
    
    parser.add_argument(
        "--debug-mode",
        type=str2bool,
        required=True,
        help="Explicitly specify 'true' or 'false' (e.g., --debug-mode=true)"
    )

    args = parser.parse_args()

    # Load environment variables in a file called .env
    load_dotenv(override=True)
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if openai_api_key:
        print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
    else:
        print("OpenAI API Key not set")

    # Register the signal handler for KeyboardInterrupt (Ctrl+C)
    signal.signal(signal.SIGINT, handle_exit_signal)

    rag = Rag(args.debug_mode)
    rag.run()
        
