import os
import urllib.request
import gzip
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
import re
import json
import time
from typing import List, Dict
from tqdm import tqdm

try:
    import nltk
    from nltk.tokenize import sent_tokenize
except Exception:
    nltk = None
    sent_tokenize = None

# ================= CONFIG =================
BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
PUBMED_PREFIX = "26"
NUM_FILES = 10
MAX_DOCS_PER_FILE = None
MAX_WORDS_PER_CHUNK = 120
OVERLAP_SENTENCES = 1

RAW_DIR = "pubmed/raw"
OUT_DIR = "pubmed/processed"
OUT_FILE = f"{OUT_DIR}/pubmed_passages_enhanced.json"
# =========================================

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


BIO_ENTITY_PATTERNS = [
    r"\b(?:BRCA1|BRCA2|TP53|EGFR|TNF|IL-6)\b",
    r"\b(?:Parkinson(?:'s)? disease|Alzheimer(?:'s)? disease|cancer|diabetes)\b",
    r"\b(?:aspirin|ibuprofen|metformin|statin|alpha-bisabolol)\b",
]


# 1. Clean raw text to remove extra whitespace and special characters
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,%()\-:; /]", "", text)
    return text.strip()


# 2. Download NLTK tokenizer if not already present
def maybe_download_punkt() -> None:
    if not nltk or not sent_tokenize:
        return
    try:
        sent_tokenize("Sentence one. Sentence two.")
    except LookupError:
        nltk.download("punkt", quiet=True)


# 3. Split text into individual sentences
def split_sentences(text: str) -> List[str]:
    if sent_tokenize:
        try:
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except LookupError:
            maybe_download_punkt()
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
    # Fallback if nltk is unavailable.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


# 4. Search for specific biomedical terms in text
def extract_entities(text: str) -> List[str]:
    found = []
    for pattern in BIO_ENTITY_PATTERNS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for m in matches:
            normalized = m.strip()
            if normalized and normalized.lower() not in {e.lower() for e in found}:
                found.append(normalized)
    return found


# 5. Group sentences into fixed-size passages (chunks) with overlap
def chunk_sentences(sentences: List[str], max_words: int, overlap_sentences: int):
    chunk = []
    chunk_words = 0

    for sentence in sentences:
        sent_words = len(sentence.split())
        if chunk and chunk_words + sent_words > max_words:
            yield " ".join(chunk)
            if overlap_sentences > 0:
                chunk = chunk[-overlap_sentences:]
                chunk_words = sum(len(s.split()) for s in chunk)
            else:
                chunk = []
                chunk_words = 0

        chunk.append(sentence)
        chunk_words += sent_words

    if chunk:
        yield " ".join(chunk)


# 6. Download a single .gz file from the PubMed FTP
def download_file(filename: str, retries: int = 3) -> str:
    path = os.path.join(RAW_DIR, filename)
    for attempt in range(1, retries + 1):
        try:
            if not os.path.exists(path):
                urllib.request.urlretrieve(BASE_URL + filename, path)
            if os.path.getsize(path) == 0:
                raise IOError(f"Downloaded empty file: {filename}")
            return path
        except Exception:
            if os.path.exists(path):
                os.remove(path)
            if attempt == retries:
                raise
            time.sleep(attempt)
    return path


# 7. Extract the .gz file to a raw .xml file
def extract_gz(gz_path: str, source_filename: str, retries: int = 3) -> str:
    xml_path = gz_path.replace(".gz", "")
    # Rebuild XML if a previous run left an empty/bad file.
    if os.path.exists(xml_path) and os.path.getsize(xml_path) > 0:
        return xml_path
    if os.path.exists(xml_path):
        os.remove(xml_path)

    for attempt in range(1, retries + 1):
        tmp_xml = f"{xml_path}.tmp"
        try:
            with gzip.open(gz_path, "rb") as f_in:
                with open(tmp_xml, "wb") as f_out:
                    while True:
                        chunk = f_in.read(1024 * 1024)
                        if not chunk:
                            break
                        f_out.write(chunk)
            os.replace(tmp_xml, xml_path)
            return xml_path
        except (EOFError, OSError):
            if os.path.exists(tmp_xml):
                os.remove(tmp_xml)
            if os.path.exists(gz_path):
                os.remove(gz_path)
            if os.path.exists(xml_path):
                os.remove(xml_path)
            if attempt == retries:
                raise
            gz_path = download_file(source_filename, retries=1)
            time.sleep(attempt)
    return xml_path


# 8. Parse the XML file to extract titles, abstracts, and create passages
def parse_xml(xml_file: str, pid_start: int):
    passages: List[Dict] = []
    pid = pid_start
    context = ET.iterparse(xml_file, events=("end",))

    for _, elem in context:
        if elem.tag != "PubmedArticle":
            continue

        pmid = elem.findtext(".//PMID")
        title = elem.findtext(".//ArticleTitle")
        abstracts = elem.findall(".//AbstractText")
        abstract = " ".join(a.text for a in abstracts if a.text)

        if title and abstract:
            full_text = clean_text(f"{title}. {abstract}")
            entities = extract_entities(full_text)
            sentences = split_sentences(full_text)

            for chunk in chunk_sentences(
                sentences,
                max_words=MAX_WORDS_PER_CHUNK,
                overlap_sentences=OVERLAP_SENTENCES,
            ):
                chunk_entities = extract_entities(chunk)
                passages.append(
                    {
                        "passage_id": f"pubmed_{pid}",
                        "pmid": pmid,
                        "text": chunk,
                        "entities": chunk_entities if chunk_entities else entities,
                    }
                )
                pid += 1

        elem.clear()
        if MAX_DOCS_PER_FILE and len(passages) >= MAX_DOCS_PER_FILE:
            break

    return passages, pid


# ================= MAIN PIPELINE =================
# A. Ensure tokenizer is ready
maybe_download_punkt()
passages = []
pid = 0

# B. Iterate through the target number of PubMed files
for i in tqdm(range(1, NUM_FILES + 1), desc="PubMed files"):
    fname = f"pubmed{PUBMED_PREFIX}n{i:04d}.xml.gz"
    print(f"Processing {fname}")

    file_done = False
    for attempt in range(1, 4):
        try:
            # Step 1: Download
            gz_path = download_file(fname, retries=3)
            # Step 2: Unzip
            xml_path = extract_gz(gz_path, fname, retries=3)
            # Step 3: Parse and Chunk
            new_passages, pid = parse_xml(xml_path, pid)
            passages.extend(new_passages)
            file_done = True
            break
        except ParseError:
            # Bad XML -> force clean re-download/extract and retry.
            gz_path = os.path.join(RAW_DIR, fname)
            xml_path = gz_path.replace(".gz", "")
            if os.path.exists(xml_path):
                os.remove(xml_path)
            if os.path.exists(gz_path):
                os.remove(gz_path)
            if attempt == 3:
                print(f"Skipping {fname} after repeated XML parse failures")
        except Exception:
            if attempt == 3:
                raise

    if not file_done:
        continue

# C. Save all processed passages to a single JSON file
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(passages, f, indent=2)

print(f"DONE: {len(passages)} passages created -> {OUT_FILE}")
