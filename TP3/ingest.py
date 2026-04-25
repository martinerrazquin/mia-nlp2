from pathlib import Path
import time
from itertools import batched
from dotenv import load_dotenv
from pinecone import Pinecone
import os

from langchain_text_splitters import MarkdownHeaderTextSplitter

load_dotenv()

EMBEDDING_MODEL = "multilingual-e5-large"

docs_path = Path("./docs")

def chunk_one(text: str) -> list[dict]:
    """
    Split one markdown-formatted text into multiple chunks, by subsections (#, ##).
    Each chunk is a dictionary with 'metadata' and 'text' fields.
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]

    # MD splits
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )

    docs = markdown_splitter.split_text(text)

    return [{'_id':f"v{idx+1}", 'text': doc.page_content} for idx, doc in enumerate(docs)]


if __name__ == '__main__':
    # get all cvs + quick health checks
    cvs = list(docs_path.iterdir())
    assert all((p.is_file() and p.suffix.lower()=='.md') for p in cvs), "docs/ should only contain Markdown files."
    assert len(cvs) > 0, "docs/ is empty"

    # init pinecone + early health checks
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


    # chunk cvs
    chunks = []
    for idx, cv in enumerate(cvs):
        # chunk one
        cur_chunks = chunk_one(cv.read_text())

        print(f"[{idx+1:>2}/{len(cvs)}] {cv.stem} -> {len(cur_chunks)} chunks...", end="")

        chunks.extend(cur_chunks)

        # create index by copying file name and replacing underscores with hyphens
        # e.g. martin_errazquin.md -> martin-errazquin
        index_name = cv.stem.replace("_","-")

        if not pc.has_index(index_name):
            pc.create_index_for_model(
                name=index_name,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model":EMBEDDING_MODEL,
                    "field_map":{"text": "text"}
                }
            )

        # can upload up to 96 text records per batch, so round it down to 90
        # https://docs.pinecone.io/guides/index-data/upsert-data#upsert-limits
        batch_size = 90
        for batch in batched(chunks, batch_size):
            pc.Index(index_name).upsert_records(os.getenv("PINECONE_INDEX_NAMESPACE"), list(batch))
            time.sleep(2)   # sleep between uploads to prevent overloading

        print("uploaded.")

    print("\nAll done.")
