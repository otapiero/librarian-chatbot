from langchain.prompts import PromptTemplate
import openai
from dotenv import load_dotenv
import os

load_dotenv()


def extract_book_description(conversation: str) -> str:
    """
    Extract the book description from the summary.
    """
    # template = """given the conversation between a costumer and a Librarian,
    #        I want you to summarize the conversation to a book description that the costumer will find interesting based on the conversation.
    #        do not include the conversation in the book description.
    #        and do not mention the costumer or the Librarian in the book description.
    #        conversation:
    #        {conversation}
    #        """

    # prompt_template = PromptTemplate(input_variables=["conversation"], template=template)
    # prompt = prompt_template.format_prompt(conversation=conversation).to_string()
    prompt = conversation
    tools = [
        {
            "name": "get_book_from_description",
            "description": "get a book by the description from the summary",
            "parameters": {
                "type": "object",
                "properties": {
                    "genres": {
                        "type": "string",
                        "description": "All the genres that will much the book description",
                    },
                    "summary": {
                        "type": "string",
                        "description": "A long and well describe book description",
                    },
                },
                "required": ["genres", "summary"],
            },
        }
    ]
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        functions=tools,
        function_call={"name": "get_book_from_description"},
    )

    return response.choices[0].message["function_call"].arguments


if __name__ == "__main__":
    conversation = """
    Customer: Good afternoon! I'm looking for a good book to read, something engaging and thought-provoking. Any recommendations?

Librarian: Good afternoon! I'd be delighted to help you find the perfect book. To better assist you, could you share more about your reading preferences? Do you lean towards fiction or non-fiction? Are there specific genres or themes you enjoy?

Customer: I'm open to both fiction and non-fiction. Recently, I've been into historical fiction, but I'm also interested in thought-provoking non-fiction, particularly in science or philosophy.

Librarian: Wonderful choices! Historical fiction can offer a rich narrative, and delving into science or philosophy can be intellectually rewarding. Can you provide more details about what you enjoyed in the historical fiction you've read? Any specific time periods or themes that stood out?

Customer: I loved the depth of characters and the historical setting in Ken Follett's "The Pillars of the Earth." I'm generally drawn to well-developed characters within historical contexts. As for non-fiction, I'm interested in the history of scientific discoveries and philosophical ideas.

Librarian: Excellent! It sounds like you appreciate character-driven narratives in historical fiction and have a keen interest in the history of science and philosophy. Are there any specific aspects or sub-genres within historical fiction that you find particularly intriguing? Additionally, do you prefer a more immersive writing style, or are you open to different narrative approaches?

Customer: I enjoy the medieval setting in historical fiction, and a writing style that immerses me in the time period is definitely a plus.

Librarian: Thank you for sharing that. A medieval setting and immersive writing style provide valuable insights. Is there anything else you'd like to add about your reading preferences, perhaps related to pacing, tone, or specific historical events that captivate you?

Customer: I think I prefer a moderate pacing, not too fast or slow, and a tone that balances drama with some moments of reflection. Specific historical events like political intrigues or cultural shifts do capture my interest.

Librarian: Your preferences are crystal clear. A balanced pacing, immersive tone, and a focus on historical events such as political intrigues align well with your interests. I'll use this information to help you find the perfect book. Is there anything else you'd like to share before we explore some potential recommendations?

Customer: No, I think that covers it. Thank you for your help!

Librarian: My pleasure! With these details, I'll be able to narrow down some excellent options for you. If you have any more questions or if there's anything specific you're hoping to find in a book, feel free to let me know. We're here to make sure you have a great reading experience!  
    """
    book_description = extract_book_description(conversation)
    from recommend_books_bot.similarity_search import find_similar_books

    results = find_similar_books(book_description)
    print(results)

    print("done")
