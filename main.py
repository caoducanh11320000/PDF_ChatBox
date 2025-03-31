from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions  # Các hàm tạo embedding của ChromaDB
import uuid
import gradio as gr
from openai import OpenAI


def load_pdf_and_create_chunks(path_document):
    loader = PyMuPDFLoader(path_document)
    data = loader.load()
    chunker = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )
    chunks = chunker.split_documents(data)
    return chunks


COLLECTION_NAME = "test_read_pdf"
message_history = []
client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key="d33017896c9d0aa9d60f0a55fae1f2b9d7715ce9e17f7014d1f049ea95b77652"
)


def create_collection(chunks):
    client = chromadb.PersistentClient(path="./data")
    collection = client.get_or_create_collection(name=COLLECTION_NAME,
                                                 embedding_function=embedding_functions.DefaultEmbeddingFunction())
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]

    collection.add(
        documents=[doc.page_content for doc in chunks],
        metadatas=[doc.metadata for doc in chunks],
        ids=ids
    )
    print(f"Tong so element in collection: {collection.count()}")

    return collection


def query_text(collection, query):
    results = collection.query(query_texts=query, n_results=3)
    print("-------------------------------")
    print(results["ids"])
    print(results["documents"])
    print(results["metadatas"])
    return results


import base64


def create_pdf_html(file_path: str | None):
    """Tạo HTML để xem trước PDF"""
    if file_path is None:
        return ""

    try:
        # Đọc file PDF
        with open(file_path.name, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')

        # Tạo HTML với PDF viewer nhúng
        pdf_display = f'''
            <div style="width: 100%; height: 800px;">
                <iframe
                    src="data:application/pdf;base64,{base64_pdf}"
                    width="100%"
                    height="100%"
                    style="border: none;">
                </iframe>
            </div>
        '''
        return pdf_display
    except Exception as e:
        return f"Error displaying PDF: {str(e)}"


def process_file(file_path: str | None):
    """Xử lý file PDF đã tải lên và trả về trạng thái cùng bản xem trước HTML"""

    if file_path is None:
        yield "Please upload a PDF file first.", ""

    global collection_total
    try:
        # Tạo bản xem trước PDF
        yield "Reading PDF file...", "", None
        pdf_html = create_pdf_html(file_path)
        yield "Processing PDF file...", pdf_html, None
        chunks = load_pdf_and_create_chunks(file_path)
        collection = create_collection(chunks)
        yield "Processing PDF is Done", pdf_html, collection
    except Exception as e:
        yield f"Error processing file: {str(e)}", "", None


def create_graphic():
    with gr.Blocks() as demo:
        gr.Markdown("## Chat with PDF")
        with gr.Row():
            with gr.Column(scale=1):
                pdf_file = gr.File(label="", file_types=[".pdf"])
                but_process = gr.Button("Process Document")
                data_process_status = gr.Textbox(label="Document status")
                html = gr.HTML()
                collection_total = gr.State()

            with gr.Column(scale=2):
                text_query = gr.Textbox(label="What is your question")
                but_search = gr.Button("Search")
                chatBox = gr.Chatbot()

            but_process.click(fn=process_file, inputs=[pdf_file], outputs=[data_process_status, html, collection_total])
            but_search.click(fn=respond_from_openai, inputs=[text_query, collection_total],
                             outputs=[text_query, chatBox])

        return demo


def respond_from_openai(question, collection):
    rag = query_text(collection=collection, query=question)

    prompt = f""""
    Use the following CONTEXT to answer the QUESTION at the end.
    If you don't know the answer or unsure of the answer, just say that you don't know, don't try to make up an answer.
    Use an unbiased and journalistic tone.

    CONTEXT = {question}

    QUESTION = {rag}
    """""

    message = []
    for mes in message_history:
        message.append({"role": "user", "content": mes[0]})
        message.append({"role": "assistant", "content": mes[1]})

    message.append({"role": "user", "content": prompt})
    message_history.append([question, "Đang xử lý..."])
    yield question, message_history
    response = client.chat.completions.create(messages=message, model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                                              stream=True)

    message_history[-1][1] = ""
    for chunk in response:
        delta = chunk.choices[0].delta.content or ""
        message_history[-1][1] += delta
        yield question, message_history

    yield "", message_history


demo = create_graphic()
demo.launch()

