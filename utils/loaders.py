import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

def load_webpage(url: str) -> str:
    loader = WebBaseLoader(
        web_paths = (url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
        )
    ),
)
    
    docs = loader.load()

    # safety filter
    docs = [d for d in docs if d.page_content.strip()]
    
    
    return docs


def load_text(text: str) -> str:
    return [Document(page_context=text)]

