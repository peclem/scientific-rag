"""
GROBID client. GROBID turns a scientific PDF into structured TEI XML, which
is far better than raw text extraction: it separates the body from the
references, recovers section headings, and gives us clean header metadata
(title, authors, date). We lean on it for structure and bibliographic info.

GROBID runs as a Docker container (see scripts/start_grobid.sh). If it isn't
reachable, the higher-level parser falls back to PyMuPDF, so nothing here
should hard-crash the pipeline; it raises and lets the caller decide.

Page numbers: the default fulltext output does not carry reliable page
positions, so blocks come back with page=None here. We fill pages later by
matching block text against PyMuPDF's per-page text (see page_mapper.py).
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
from loguru import logger
from lxml import etree

from rag.ingestion.schema import ParsedDocument, TextBlock

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def is_grobid_alive(base_url: str, timeout_s: float = 5.0) -> bool:
    """Cheap health check so the caller can decide whether to fall back."""
    try:
        r = httpx.get(f"{base_url}/api/isalive", timeout=timeout_s)
        return r.status_code == 200 and "true" in r.text.lower()
    except Exception as exc:  # noqa: BLE001 - any failure means "not usable"
        logger.debug(f"GROBID not alive at {base_url}: {exc}")
        return False


def parse_pdf_grobid(pdf_path: str | Path, base_url: str, timeout_s: int = 60) -> ParsedDocument:
    """Send a PDF to GROBID and turn the TEI response into a ParsedDocument."""
    pdf_path = Path(pdf_path)
    with pdf_path.open("rb") as fh:
        files = {"input": (pdf_path.name, fh, "application/pdf")}
        # consolidateHeader=1 lets GROBID clean header metadata against
        # CrossRef when it can, which improves author/year quality.
        data = {"consolidateHeader": "1", "consolidateCitations": "0"}
        resp = httpx.post(
            f"{base_url}/api/processFulltextDocument",
            files=files,
            data=data,
            timeout=timeout_s,
        )
    resp.raise_for_status()
    return _parse_tei(resp.content, str(pdf_path))


def _text(node) -> str:
    """All inner text of an element, whitespace-normalized."""
    if node is None:
        return ""
    return re.sub(r"\s+", " ", "".join(node.itertext())).strip()


def _parse_tei(tei_bytes: bytes, source_path: str) -> ParsedDocument:
    root = etree.fromstring(tei_bytes)

    # --- Header metadata ---
    title_el = root.find(".//tei:teiHeader//tei:titleStmt/tei:title", TEI_NS)
    title = _text(title_el) or None

    authors: list[str] = []
    for pers in root.findall(
        ".//tei:teiHeader//tei:sourceDesc//tei:biblStruct//tei:author/tei:persName",
        TEI_NS,
    ):
        forenames = [_text(f) for f in pers.findall("tei:forename", TEI_NS)]
        surname = _text(pers.find("tei:surname", TEI_NS))
        full = " ".join([*[f for f in forenames if f], surname]).strip()
        if full:
            authors.append(full)

    year = None
    for date_el in root.findall(".//tei:teiHeader//tei:date", TEI_NS):
        when = date_el.get("when", "")
        m = re.search(r"(19|20)\d{2}", when)
        if m:
            year = int(m.group(0))
            break

    blocks: list[TextBlock] = []

    # --- Abstract ---
    for p in root.findall(".//tei:profileDesc/tei:abstract//tei:p", TEI_NS):
        txt = _text(p)
        if txt:
            blocks.append(TextBlock(text=txt, section="Abstract"))

    # --- Body, section by section ---
    for div in root.findall(".//tei:text/tei:body/tei:div", TEI_NS):
        head = _text(div.find("tei:head", TEI_NS)) or None
        for p in div.findall("tei:p", TEI_NS):
            txt = _text(p)
            if txt:
                blocks.append(TextBlock(text=txt, section=head))

    doc = ParsedDocument(
        source_path=source_path,
        parser="grobid",
        title=title,
        authors=authors,
        year=year,
        blocks=blocks,
    )
    logger.info(
        f"GROBID parsed '{pdf_path_name(source_path)}': "
        f"{len(blocks)} blocks, bibkey={doc.bibkey}, "
        f"title={'yes' if title else 'no'}, authors={len(authors)}"
    )
    return doc


def pdf_path_name(path: str) -> str:
    return path.rsplit("/", 1)[-1]
