"""
Standalone test script: extract text from a PDF and POST it to /api/ingest.

Does NOT modify ingest.sh or any app code. Run it, see if ingestion
succeeds, then decide whether to keep this as a real feature.

Usage:
    python test_pdf_ingest.py agent_skills.pdf
"""
import sys
import json
import urllib.request

try:
    from pypdf import PdfReader
except ImportError:
    sys.exit("pypdf not installed. Run: pip install pypdf")

API_URL = "http://localhost:8000/api/ingest"


def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def main():
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python {sys.argv[0]} <file.pdf>")

    pdf_path = sys.argv[1]
    text = extract_text(pdf_path)
    print(f"Extracted {len(text)} chars from {pdf_path}")

    if not text.strip():
        sys.exit("No text extracted (PDF may be scanned/image-based).")

    payload = {
        "documents": [
            {
                "title": pdf_path,
                "content": text,
                "metadata": {"source": "local-file"},
            }
        ],
        "chunk_size": 500,
        "chunk_overlap": 50,
        "replace_existing": True,
    }

    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            print(f"Status: {resp.status}")
            print(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"Status: {e.code}")
        print(e.read().decode())


if __name__ == "__main__":
    main()
