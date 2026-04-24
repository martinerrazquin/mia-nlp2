from pathlib import Path
import time
from tqdm import tqdm
from itertools import batched
from dotenv import load_dotenv
from pinecone import Pinecone
import os

from langchain_text_splitters import MarkdownHeaderTextSplitter

load_dotenv()


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

    return [{'text': doc.page_content} for doc in docs]


if __name__ == '__main__':
    # get all cvs + quick health checks
    cvs = list(docs_path.iterdir())
    assert all((p.is_file() and p.suffix.lower()=='.md') for p in cvs), "docs/ should only contain Markdown files."
    assert len(cvs) > 0, "docs/ is empty"

    # init pinecone + early health checks
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    index_name = os.getenv("PINECONE_INDEX_NAME")
    index = pc.Index(index_name)
    assert index_name in pc.list_indexes().names(), "Pinecone index name does not exist"


    # chunk cvs
    chunks = []
    for idx, cv in enumerate(cvs):
        cur_chunks = chunk_one(cv.read_text())

        print(f"[{idx+1:>2}/{len(cvs)}] {cv.stem} -> {len(cur_chunks)} chunks")

        chunks.extend(cur_chunks)

    # add '_id' field for pinecone
    for idx, chunk in enumerate(chunks):
        chunk['_id'] = f"v{idx+1}"

    
    # can upload up to 96 text records per batch, so round it down to 90
    # https://docs.pinecone.io/guides/index-data/upsert-data#upsert-limits
    batch_size = 90
    for batch in batched(chunks, batch_size):
        index.upsert_records(os.getenv("PINECONE_INDEX_NAMESPACE"), list(batch))
        time.sleep(2)   # sleep between uploads to prevent overloading

    print("All done.")
