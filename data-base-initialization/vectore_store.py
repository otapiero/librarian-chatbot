import logging
import json
import csv
import os

import pinecone
import dotenv

from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import CSVLoader
import pandas as pd
from pandas import DataFrame

from consts import INDEX_NAME, BOOKS_CSV_PATH


dotenv.load_dotenv()
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def load_books_data() -> DataFrame:
    data = []
    # open with utf8 encoding since there are some special characters in the dataset
    with open(r"../" + BOOKS_CSV_PATH, encoding="utf8") as f:
        reader = csv.reader(f, dialect="excel-tab")
        for row in reader:
            data.append(row)

    # convert data to pandas dataframe
    books = pd.DataFrame.from_records(
        data,
        columns=[
            "book_id",
            "freebase_id",
            "book_title",
            "author",
            "publication_date",
            "genre",
            "summary",
        ],
    )
    books["genre"] = books["genre"].apply(parse_genre_entry)
    return books


def parse_genre_entry(genre_info):
    if genre_info == "":
        return []
    genre_dict = json.loads(genre_info)
    genres = list(genre_dict.values())
    return genres


def transform_data_to_csv():
    books = load_books_data()
    # save to csv
    books.to_csv("../data/books.csv", index=False)


def filter_too_long_documents(csv_document):
    lst = []
    for doc in csv_document:
        if len(doc.page_content) < 9000:
            lst.append(doc)
    return lst


def ingest_data():
    transform_data_to_csv()
    loader = CSVLoader("../data/books.csv", encoding="utf8")
    csv_document = loader.load()
    csv_document = filter_too_long_documents(csv_document)
    # peek the top 5000 documents

    csv_document = csv_document[:5000]
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    logging.info("Creating index")
    Pinecone.from_documents(
        documents=csv_document, embedding=embeddings, index_name=INDEX_NAME
    )
    logging.info("Done ingesting documents")


if __name__ == "__main__":
    logging.basicConfig(
        filename="../logs/ingest_data.log",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    ingest_data()
