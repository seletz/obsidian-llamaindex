import sys
import logging.config
import os.path
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    load_index_from_storage,
)
from llama_index import download_loader
from llama_index.indices.base import BaseIndex
from llama_index.llms import Ollama

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def setup_logging() -> None:
    if os.path.exists("logging.conf"):
        logging.config.fileConfig("logging.conf", encoding="UTF-8")
        logger.info("Configured logging system from file.")
    else:
        logger.warning("No logging config found")
    logger.setLevel(logging.DEBUG)


def index_data(path: str, index_path: str) -> VectorStoreIndex:
    path = os.path.expanduser(path)
    ObsidianReader = download_loader('ObsidianReader')

    # load the documents and create the index
    logger.info(f"Loading documents from {path}.")
    documents = ObsidianReader(path).load_data()
    logger.info(f"Loaded {len(documents)} documents.")

    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=index_path)
    return index


def get_vector_index() -> VectorStoreIndex | BaseIndex:
    data_path = "~/develop/notes"
    index_path = "./storage"
    # check if storage already exists
    if not os.path.exists(index_path):
        logger.info("Creating index.")
        return index_data(data_path, index_path)

    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    return load_index_from_storage(storage_context)


if __name__ == "__main__":
    setup_logging()

    llm = Ollama(model="mistral")

    service_context = ServiceContext.from_defaults(llm=llm)

    index = get_vector_index()

    # either way we can now query the index
    query_engine = index.as_query_engine(service_context=service_context)
    response = query_engine.query("An was habe ich am 2023-11-15 gearbeitet?")
    print(response)
