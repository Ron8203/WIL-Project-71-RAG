from pathlib import Path
import json
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter


RAW_DIR = Path("data/raw/cyber_docs")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "chunks.json"


def extract_pdf_pages(pdf_path):
    pdf = fitz.open(pdf_path)
    pages = []

    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text("text").strip()

        # Ignore cover pages or pages with very little useful text
        if len(text) >= 200:
            pages.append(
                {
                    "source": pdf_path.name,
                    "page": page_number,
                    "text": text,
                }
            )

    pdf.close()

    return pages


def create_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for page in pages:
        split_texts = splitter.split_text(page["text"])

        for chunk_index, chunk_text in enumerate(split_texts):
            chunks.append(
                {
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_id": f'{page["source"]}_p{page["page"]}_c{chunk_index}',
                    "text": chunk_text.strip(),
                }
            )

    return chunks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    all_pages = []

    print(f"Found {len(pdf_files)} PDF files.\n")

    for pdf_file in pdf_files:
        pages = extract_pdf_pages(pdf_file)
        all_pages.extend(pages)

        print(
            f"{pdf_file.name}: "
            f"{len(pages)} pages with extracted text"
        )

    chunks = create_chunks(all_pages)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            chunks,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nProcessing complete.")
    print(f"Total pages processed: {len(all_pages)}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()