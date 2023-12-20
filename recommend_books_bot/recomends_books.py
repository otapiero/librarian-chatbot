from .similarity_search import find_similar_books
from .summarize_conversation import extract_book_description


def get_recommended_books(conversation: str)-> list:
    """
    Extract the book description from the summary.
    """
    book_description = extract_book_description(conversation)
    results = find_similar_books(book_description)
    return results
