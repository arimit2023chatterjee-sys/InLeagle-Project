import os
import re
import logging
import argparse
from pathlib import Path         
from typing import Optional

import pdfplumber                 

from dotenv import load_dotenv
load_dotenv()

from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

RAW_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Chunking
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 80
MIN_CHUNK_LEN = 80

# Qdrant
QDRANT_URL      = os.getenv("QDRANT_CLOUD_CLUSTER_URL") or os.getenv("QDRANT_URL") or "http://localhost:6333"
QDRANT_API_KEY  = os.getenv("QDRANT_CLOUD_API_KEY") or ""
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION") or "InLegalDocs"

# Embedding model
# bhavyagiri/InLegal-Sbert is a sentence-transformer variant of InLegalBERT
# output dim = 768, trained on Indian legal corpus — correct choice
MODEL_NAME = "bhavyagiri/InLegal-Sbert"

# ─────────────────────────────────────────────────────────────
# DOCUMENT REGISTRY
# ─────────────────────────────────────────────────────────────

DOCUMENTS = {

    # ── ACTS ────────────────────────────────────────────────

    "sarfaesi": {
        "path": RAW_DIR / "acts" / "sarfaesi_2002.pdf",
        "slice_from": None,
        "metadata": {
            "source_type":   "act",
            "document_name": "SARFAESI Act 2002",
            "year":          2002,
            # Qdrant payload via LangChain does not reliably handle list values
            "topic_tags": "enforcement,security interest,property seizure,NPA,"
                          "borrower rights,DRT appeal,60 day notice,possession,auction",
        },
    },

    "ni_act": {
        "path": RAW_DIR / "acts" / "ni_act_1881.pdf",
        "slice_from": ["138.", "Section 138", "138\n"],
        "metadata": {
            "source_type":   "act",
            "document_name": "Negotiable Instruments Act 1881",
            "year":          1881,
            "topic_tags": "cheque bounce,section 138,criminal liability,dishonour,"
                          "demand notice,director liability,stop payment,NI Act",
        },
    },

    "banking_regulation": {
        "path": RAW_DIR / "acts" / "banking_regulation_act_1949.pdf",
        "slice_from": None,
        "metadata": {
            "source_type":   "act",
            "document_name": "Banking Regulation Act 1949",
            "year":          1949,
            "topic_tags": "bank license,RBI inspection,account refusal,bank merger,"
                          "deposit safety,RBI powers,bank winding up,BR Act",
        },
    },

    "pmla": {
        "path": RAW_DIR / "acts" / "pmla_2002.pdf",
        "slice_from": None,
        "metadata": {
            "source_type":   "act",
            "document_name": "Prevention of Money Laundering Act 2002",
            "year":          2002,
            "topic_tags": "account freeze,money laundering,suspicious transaction,"
                          "ED attachment,FIU,provisional attachment,PMLA",
        },
    },

    "ibc": {
        "path": RAW_DIR / "acts" / "ibc_2016.pdf",
        "slice_from": None,
        "metadata": {
            "source_type":   "act",
            "document_name": "Insolvency and Bankruptcy Code 2016",
            "year":          2016,
            "topic_tags": "insolvency,CIRP,moratorium,personal guarantor,liquidation,"
                          "financial creditor,NCLT,resolution plan,Section 14,IBC",
        },
    },

    "cic": {
        "path": RAW_DIR / "acts" / "cic_act_2005.pdf",
        "slice_from": None,
        "metadata": {
            "source_type":   "act",
            "document_name": "Credit Information Companies Act 2005",
            "year":          2005,
            "topic_tags": "CIBIL,credit score,credit report,NPA wrongful tagging,"
                          "wilful defaulter,dispute credit record,CIC Act",
        },
    },

    # ── RBI DOCUMENTS ───────────────────────────────────────

    "ombudsman": {
        "path": RAW_DIR / "rbi_documents" / "ombudsman_scheme__digital_transactions_2019.pdf",
        "slice_from": None,
        "metadata": {
            "source_type":   "rbi_circular",
            "document_name": "RBI Ombudsman Scheme for Digital Transactions 2019",
            "year":          2019,
            "topic_tags": "digital transaction complaint,RBI ombudsman digital,"
                          "grievance against bank,consumer complaint bank",
        },
    },

    "digital_lending": {
        "path": RAW_DIR / "rbi_documents" / "ombudsman_scheme__non_banking_financial_companies_2018.pdf",
        "slice_from": None,
        "metadata": {
            "source_type":   "rbi_circular",
            "document_name": "RBI Ombudsman Scheme for Non-Banking Financial Companies 2018",
            "year":          2018,
            "topic_tags": "digital lending complaint,NBFC complaint,RBI ombudsman NBFC,"
                          "grievance against NBFC,consumer complaint NBFC",
        },
    },

    "fair_practices": {
        "path": RAW_DIR / "rbi_documents" / "obudsman_scheme_for_banking_2006.pdf",
        "slice_from": None,
        "metadata": {
            "source_type":   "rbi_circular",
            "document_name": "RBI Ombudsman Scheme for Banking 2006",
            "year":          2006,
            "topic_tags": "fair practices code,bank lending regulation,"
                          "loan application process,interest rate transparency,"
                          "grievance redressal,RBI complaint bank",
        },
    },

    # ── CASES ───────────────────────────────────────────────

    "mardia_chemicals": {
        "path": RAW_DIR / "cases" / "mardia_chemicals_2004.pdf",
        "slice_from": None,
        "metadata": {
            "source_type":   "case",
            "document_name": "Mardia Chemicals v Union of India 2004",
            "year":          2004,
            "court":         "Supreme Court",
            "topic_tags": "SARFAESI constitutional validity,borrower rights,"
                          "75 percent deposit DRT,constitutional challenge",
        },
    },

    "dashrath_rupsingh": {
        "path": RAW_DIR / "cases" / "dashrath_rupsingh_2014.pdf",
        "slice_from": None,
        "metadata": {
            "source_type":   "case",
            "document_name": "Dashrath Rupsingh Rathod v State of Maharashtra 2014",
            "year":          2014,
            "court":         "Supreme Court",
            "topic_tags": "cheque bounce jurisdiction,section 138,"
                          "where to file case,territorial jurisdiction",
        },
    },

    "innoventive": {
        "path": RAW_DIR / "cases" / "innoventive_industries_2017.pdf",
        "slice_from": None,
        "metadata": {
            "source_type":   "case",
            "document_name": "Innoventive Industries v ICICI Bank 2017",
            "year":          2017,
            "court":         "Supreme Court",
            "topic_tags": "IBC moratorium,CIRP Section 14,"
                          "recovery stopped insolvency,IBC overrides state law",
        },
    },

    "lalit_kumar": {
        "path": RAW_DIR / "cases" / "lalit_kumar_jain_2021.pdf",
        "slice_from": None,
        "metadata": {
            "source_type":   "case",
            "document_name": "Lalit Kumar Jain v Union of India 2021",
            "year":          2021,
            "court":         "Supreme Court",
            "topic_tags": "personal guarantor IBC,Section 95,"
                          "guarantor cannot hide behind moratorium,MSME guarantor",
        },
    },

    "meters_instruments": {
        "path": RAW_DIR / "cases" / "meters_instruments_2017.pdf",
        "slice_from": None,
        "metadata": {
            "source_type":   "case",
            "document_name": "Meters and Instruments v Kanchan Mehta 2017",
            "year":          2017,
            "court":         "Supreme Court",
            "topic_tags": "cheque bounce settlement,section 138 closure,"
                          "pay compensation close case,accused exit route",
        },
    },

    "suman_sethi": {
        "path": RAW_DIR / "cases" / "suman_sethi_2000.pdf",
        "slice_from": None,
        "metadata": {
            "source_type":   "case",
            "document_name": "Suman Sethi v Ajay K Churiwal 2000",
            "year":          2000,
            "court":         "Supreme Court",
            "topic_tags": "stop payment not defence,section 138 liability,"
                          "cheque dishonour stop payment,criminal liability remains",
        },
    },

    "icici_shanti": {
        "path": RAW_DIR / "cases" / "icici_shanti_devi_2008.pdf",
        "slice_from": None,
        "metadata": {
            "source_type":   "case",
            "document_name": "ICICI Bank v Shanti Devi Sharma 2008",
            "year":          2008,
            "court":         "High Court",
            "topic_tags": "unauthorized transaction,bank negligence liability,"
                          "fraud bank liable,customer protection,unauthorized debit",
        },
    },
}

# ─────────────────────────────────────────────────────────────
# PARSE FUNCTIONS
# ─────────────────────────────────────────────────────────────

def parse_pdf(path: Path) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:    
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    text = _clean_text(text)

    if len(text) < 300:
        raise ValueError(
            f"Only {len(text)} chars extracted from {path.name}.\n"
            f"  → File may be a scanned image PDF (needs OCR).\n"
            f"  → Try downloading a text-based version from IndiaCode/RBI."
        )
    return text

def parse_txt(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return _clean_text(text)


def _clean_text(text: str) -> str:
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'[^\x09\x0A\x20-\x7E]', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def slice_from_keyword(text: str, keywords: list) -> str:
    for kw in keywords:
        idx = text.find(kw)
        if idx != -1:
            log.info(f"  Sliced from keyword '{kw}' at char {idx:,}")
            return text[idx:]
    log.warning(f"  None of {keywords} found — using full text")
    return text

def parse_document(doc_key: str, doc_config: dict) -> Optional[str]:
    path: Path = doc_config["path"]

    if not path.exists():
        log.error(f"File not found: {path}")
        return None

    log.info(f"Parsing: {path.name}  ({path.stat().st_size / 1024:.0f} KB)")

    try:
        ext = path.suffix.lower()
        if ext == ".pdf":
            text = parse_pdf(path)
        elif ext in (".txt", ".text"):
            text = parse_txt(path)
        else:
            log.error(f"Unsupported file type: {ext}")
            return None
    except Exception as e:
        log.error(f"Parse failed for {path.name}: {e}")
        return None

    if doc_config.get("slice_from"):
        text = slice_from_keyword(text, doc_config["slice_from"])

    log.info(f"  Extracted {len(text):,} characters")

    # Saving processed text for inspection/debugging
    source_type = doc_config["metadata"]["source_type"]
    save_dir = PROCESSED_DIR / (
        "acts"           if source_type == "act"
        else "rbi_documents" if source_type == "rbi_circular"
        else "cases"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"{doc_key}.txt", "w", encoding="utf-8") as f:
        f.write(text)

    return text

# ─────────────────────────────────────────────────────────────
# CHUNK  →  LangChain Document objects
# ─────────────────────────────────────────────────────────────

def chunk_document(text: str, metadata: dict) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
        length_function=len,
    )

    raw_chunks = splitter.split_text(text)

    docs = []
    skipped = 0
    for i, chunk in enumerate(raw_chunks):
        chunk = chunk.strip()
        if len(chunk) < MIN_CHUNK_LEN:
            skipped += 1
            continue
        docs.append(Document(
            page_content=chunk,
            metadata={
                **metadata,
                "chunk_index": i,
                "chunk_total": len(raw_chunks),
            },
        ))

    log.info(f"  Chunked → {len(docs)} documents  (skipped {skipped} short fragments)")
    return docs

# ─────────────────────────────────────────────────────────────
# EMBEDDING MODEL
# ─────────────────────────────────────────────────────────────

def load_embeddings() -> HuggingFaceEmbeddings:
    log.info(f"Loading embedding model: {MODEL_NAME}")
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True    
        },                                  
    )

# ─────────────────────────────────────────────────────────────
# INGEST ONE DOCUMENT
# ─────────────────────────────────────────────────────────────

def ingest_one(
    doc_key: str,
    doc_config: dict,
    embeddings: HuggingFaceEmbeddings,
) -> dict:
    doc_name = doc_config["metadata"]["document_name"]
    print(f"\n{'─' * 60}")
    log.info(f"[{doc_key.upper()}] {doc_name}")
    print(f"{'─' * 60}")

    text = parse_document(doc_key, doc_config)
    if text is None:
        return {"doc_key": doc_key, "status": "failed", "stage": "parse"}

    lc_docs = chunk_document(text, doc_config["metadata"])
    if not lc_docs:
        return {"doc_key": doc_key, "status": "failed", "stage": "chunk"}

    log.info(f"  Embedding and storing {len(lc_docs)} chunks in Qdrant...")
    QdrantVectorStore.from_documents(
        documents=lc_docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
        prefer_grpc=False,
        timeout=60, 
    )

    log.info(f" Done — {len(lc_docs)} chunks stored")
    return {
        "doc_key": doc_key,
        "status":  "success",
        "chars":   len(text),
        "chunks":  len(lc_docs),
    }

# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def process_all_documents(target_keys: list[str]):
    embeddings = load_embeddings()
    log.info("Embedding model ready\n")

    results = []
    for key in target_keys:
        if key not in DOCUMENTS:
            log.error(f"Unknown key '{key}' — skipping")
            continue
        r = ingest_one(key, DOCUMENTS[key], embeddings)
        results.append(r)

    print("\n" + "=" * 60)
    print("  INGESTION SUMMARY")
    print("=" * 60)

    ok  = [r for r in results if r["status"] == "success"]
    bad = [r for r in results if r["status"] != "success"]

    for r in ok:
        print(f"  Done: {r['doc_key']:<22}  {r['chunks']:>5} chunks  "
              f"{r['chars']:>8,} chars")
    for r in bad:
        print(f" Can't be Done: {r['doc_key']:<22}  failed at: {r.get('stage', '?')}")

    print(f"\n  Processed : {len(ok)}/{len(results)} documents")
    print("=" * 60)

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ingest Indian banking law documents into Qdrant"
    )
    parser.add_argument(
        "--doc",
        type=str,
        default=None,
        help=f"Ingest single doc by key. Options: {', '.join(DOCUMENTS.keys())}",
    )
    args = parser.parse_args()

    for sub in ["acts", "rbi_documents", "cases"]:
        (PROCESSED_DIR / sub).mkdir(parents=True, exist_ok=True)

    if args.doc:
        if args.doc not in DOCUMENTS:
            log.error(f"Unknown key '{args.doc}'. Available: {list(DOCUMENTS.keys())}")
            raise SystemExit(1)
        target_keys = [args.doc]
    else:
        target_keys = list(DOCUMENTS.keys())

    log.info(f"Documents to ingest: {len(target_keys)}")
    process_all_documents(target_keys)

if __name__ == "__main__":
    main()