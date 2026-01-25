import os
import urllib.request
import gzip
import xml.etree.ElementTree as ET
import re
import json
from tqdm import tqdm

# ================= CONFIG =================
BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
NUM_FILES = 10        # ⬅️ increase later (25, 50, 100...)
MAX_DOCS_PER_FILE = None
WORDS_PER_PASSAGE = 120

RAW_DIR = "pubmed/raw"
OUT_DIR = "pubmed/processed"
OUT_FILE = f"{OUT_DIR}/pubmed_passages.json"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
# ==========================================

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,%() -]", "", text)
    return text.strip()

def chunk_text(text, max_words):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])

def download_file(filename):
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        urllib.request.urlretrieve(BASE_URL + filename, path)

def extract_gz(gz_path):
    xml_path = gz_path.replace(".gz", "")
    if not os.path.exists(xml_path):
        with gzip.open(gz_path, "rb") as f_in:
            with open(xml_path, "wb") as f_out:
                f_out.write(f_in.read())
    return xml_path

def parse_xml(xml_file, pid_start):
    passages = []
    pid = pid_start

    context = ET.iterparse(xml_file, events=("end",))

    for _, elem in context:
        if elem.tag == "PubmedArticle":
            pmid = elem.findtext(".//PMID")
            title = elem.findtext(".//ArticleTitle")

            abstracts = elem.findall(".//AbstractText")
            abstract = " ".join(a.text for a in abstracts if a.text)

            if title and abstract:
                text = clean_text(title + " " + abstract)
                for chunk in chunk_text(text, WORDS_PER_PASSAGE):
                    passages.append({
                        "passage_id": f"pubmed_{pid}",
                        "pmid": pmid,
                        "text": chunk
                    })
                    pid += 1

            elem.clear()

            if MAX_DOCS_PER_FILE and len(passages) >= MAX_DOCS_PER_FILE:
                break

    return passages, pid

# ================= PIPELINE =================

passages = []
pid = 0

for i in tqdm(range(1, NUM_FILES + 1)):
    fname = f"pubmed25n{i:04d}.xml.gz"

    print(f"\n⬇ Processing {fname}")

    download_file(fname)
    gz_path = os.path.join(RAW_DIR, fname)
    xml_path = extract_gz(gz_path)

    new_passages, pid = parse_xml(xml_path, pid)
    passages.extend(new_passages)

with open(OUT_FILE, "w") as f:
    json.dump(passages, f, indent=2)

print(f"\n✅ DONE: {len(passages)} passages created")
