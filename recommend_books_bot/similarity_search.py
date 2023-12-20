import logging

import dotenv
import pinecone
import os


from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings


from consts import INDEX_NAME

dotenv.load_dotenv()
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def load_vector_store():
    logging.info("Loading vector store...")
    embeddings = OpenAIEmbeddings()
    index_name = INDEX_NAME
    return Pinecone.from_existing_index(index_name, embeddings)


def find_similar_books(book_description: str, top_k: int = 5):
    """
    Find similar books to a given book description.
    """
    logging.info("Searching for similar books...")
    docsearch = load_vector_store()
    results = docsearch.similarity_search(book_description, k=top_k)
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    # book_description = 'The ideal book for the reader is likely to be a historical fiction novel set in medieval times, featuring a rich narrative with well-developed characters akin to Ken Follett\'s "The Pillars of the Earth." The reader prefers a balanced pacing and an immersive writing style that delves into the historical context, with a tone that seamlessly combines drama and reflective moments. The narrative could explore specific historical events, such as political intrigues and cultural shifts, and may incorporate elements of philosophical and scientific ideas. This book would offer an engaging and thought-provoking experience, seamlessly blending the allure of historical settings with the intellectual depth of scientific and philosophical exploration.'
    book_description = """This book is perfect for readers who appreciate character-driven narratives in historical fiction and have a keen interest in the history of science and philosophy. Set in a medieval setting, the book offers an immersive writing style with a balanced pacing and tone that balances drama with moments of reflection. It focuses on historical events such as political intrigues and cultural shifts, providing valuable insights into the past.
"""
    results = find_similar_books(book_description)
    print(results)
