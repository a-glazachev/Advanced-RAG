"""
Module: main.py
Description: 
    This module demonstrates how to use LLamaIndex for parsing documents, storing Qqrant as vector storage, and LangChain as llm chat pipeline.
"""

import os
import tempfile
import time

import gradio as gr
import nltk
import qdrant_client
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_ollama import ChatOllama
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.file import EpubReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pymupdf4llm import LlamaMarkdownReader

from src.parser.MultiTagHTMLReader import MultiTagHTMLReader

# data directory
DATA = "../data"

# setup embedding model
ollama_embedding = OllamaEmbedding(
    model_name="qwen3-embedding:0.6b",
    base_url="http://localhost:11434",
    embed_batch_size=32
)
Settings.embed_model = ollama_embedding

# setup langchain model
llm = ChatOllama(model="qwen3:4b", temperature=0)

# setup qdrant as vector storage
client = qdrant_client.QdrantClient(
    host="localhost",
    port=6333
)
collection_name = "tech_documents"
vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context
)

# llama wrapper for langchain
llama_retriever = vector_index.as_retriever(similarity_top_k=5)


# wrap retriever from LlamaIndex into LangChain
def retrieve_nodes(query: str):
    return llama_retriever.retrieve(query)

# format nodes as LLM context
def format_context(nodes) -> str:
    # for debug
    print(f"=== Found chunks: {len(nodes)} ===\n")

    sources = []
    for i, node in enumerate(nodes, start=1):
        print(f"--- Chunk #{i} | Similarity Score: {node.score:.4f} ---")
        metadata = node.node.metadata
        file_name = metadata.get("file_name", "Empty")
        file_path = metadata.get("file_path", "Empty")
        page = metadata.get("page", "Empty")
        page_total = metadata.get("total_pages", "Empty")
        header = metadata.get("header_path", "Empty")
        content_llm = node.get_content(metadata_mode=MetadataMode.LLM)
        print(f"Metadata: {metadata}")
        print(f"File: {file_name}, Page: {page}/{page_total}, Header: {header}")
        print(f"FilePath: {file_path}")
        print("Text:")
        print(node.node.get_content())
        print("Text LLM:")
        print(content_llm)
        print("-" * 50 + "\n")

        sources.append(f"{content_llm}")

    return "\n\n---\n\n".join(sources)

# for debug
def log_messages(messages):
    print("=== MESSAGES TO LLM ===")
    for msg in messages.to_messages():
        print(f"[{msg.type.upper()}]: {msg.content}\n")
    return messages


# prompts
prompt_system = SystemMessagePromptTemplate.from_template(
"""
Ты ассистент архитектора. Отвечай на вопрос пользователя ТОЛЬКО опираясь на указанный контекст.
Всегда подтверждай каждый ответ ссылками на источники. Формат ссылки источника должен быть строго таким: [file_name, стр. page].
Если ответа нет в контексте, честно скажи, что не знаешь.

Контекст:
{context}
""")

prompt_user = HumanMessagePromptTemplate.from_template("{question}")

# langchain pipeline
rag_chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: format_context(x["nodes"]))
        )
        | ChatPromptTemplate.from_messages([prompt_system, prompt_user])
        | log_messages
        | llm
        | StrOutputParser()
)


# gradio: index files
def load_files(files):
    if not files:
        return "There are no files"

    with tempfile.TemporaryDirectory() as temp_dir:
        # setup extractors
        html_extractor = MultiTagHTMLReader(
            target_tags=["h1", "h2", "h3", "p", "li", "code", "blockquote", "em"],
            parser="html.parser"
        )
        reader = SimpleDirectoryReader(
            input_files=files,
            file_extractor={".pdf": LlamaMarkdownReader(), ".htm": html_extractor, ".html": html_extractor,
                            ".xhtml": html_extractor, ".epub": EpubReader()},
            recursive=False
        )

        # setup transformations
        transformations = [
            MarkdownNodeParser(),
            SentenceSplitter(chunk_size=1024, chunk_overlap=128, paragraph_separator="\n\n")
        ]

        # load documents (through the pipeline)
        print("Load documents ...")
        start_time = time.perf_counter()
        documents = reader.load_data()
        # update metadata keys
        for doc in documents:
            doc.excluded_llm_metadata_keys = ['file_type', 'file_size', 'creation_date', 'last_modified_date', 'format', 'title', 'author', 'subject', 'keywords', 'creator', 'producer', 'creationDate', 'modDate', 'trapped', 'encryption', 'total_pages']
        nodes = IngestionPipeline(transformations=transformations).run(documents=documents)
        print(nodes)
        end_time = time.perf_counter()
        print(f"Time elapsed: {end_time - start_time :.2f} сек")

        # make embeddings
        print("Make embeddings and save store ...")
        start_time = time.perf_counter()
        vector_index.insert_nodes(nodes)
        end_time = time.perf_counter()
        print(f"Time elapsed: {end_time - start_time :.2f} сек")

    return f"Processed {len(documents)} documents, created {len(nodes)} chunks"


# gradio: chat function
def answer_question(question, history):
    start_time = time.perf_counter()
    nodes = retrieve_nodes(question)
    end_time = time.perf_counter()
    print(f"Retriever time elapsed: {end_time - start_time :.2f} сек")

    start_time = time.perf_counter()
    response = rag_chain.invoke({"nodes": nodes, "question": question})
    end_time = time.perf_counter()
    print(f"LLM Chain time elapsed: {end_time - start_time :.2f} сек")

    sources = format_sources(nodes)
    return [
        {"role": "assistant", "content": response},
        {"role": "assistant", "content": sources}
    ]

# gradio: format source links
def format_sources(nodes):
    if not nodes:
        return ""

    lines = ["\n\n---\n**Sources (chunks from retriever)**"]
    for i, node in enumerate(nodes, 1):
        metadata = node.node.metadata
        file_name = metadata.get("file_name", "Empty")
        file_path = metadata.get("file_path", "Empty")
        page = metadata.get("page", "Empty")
        link = f"[{file_name}](/gradio_api/file={file_path})"
        lines.append(f"{i}. {link} (стр. {page})")

    return "\n".join(lines)

# gradio ui
with gr.Blocks(title="Advanced RAG LlamaIndex + Qdrant + LangChain") as demo:
    gr.Markdown("# 📚 Advanced RAG: LlamaIndex + Qdrant + LangChain")

    with gr.Tab("1. Load documents"):
        file_input = gr.File(
            label="Load documents",
            file_count="multiple",
            file_types=[".pdf", ".htm", ".xhtml", ".html", ".txt", ".epub"],
        )
        status_output = gr.Textbox(label="Status", interactive=False)
        upload_button = gr.Button("Load", variant="primary")
        upload_button.click(
            fn=load_files,
            inputs=[file_input],
            outputs=[status_output]
        )

    with gr.Tab("2. Chat"):
        gr.ChatInterface(
            fn=answer_question,
            textbox=gr.Textbox(placeholder="Ask questions ...", lines=2),
        )

if __name__ == "__main__":
    # prepare directories
    os.makedirs(f"{DATA}", exist_ok=True)

    # setup nltk
    nltk_dir = f"{DATA}/nltk"
    os.makedirs(nltk_dir, exist_ok=True)

    if nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_dir)

    nltk.download('stopwords', download_dir=nltk_dir)
    nltk.download('punkt', download_dir=nltk_dir)
    nltk.download('punkt_tab', download_dir=nltk_dir)

    # run ui
    demo.launch(allowed_paths=["/tmp/gradio"])
