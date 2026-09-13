import hashlib
import logging
import os
import re
from typing import Callable, Optional

from langchain_core.documents import Document

from src.services.llm import PaperProfile, extract_paper_profile, get_llm

logger = logging.getLogger(__name__)


def _build_granite_tokenizer():
    """Construct a Docling-compatible tokenizer backed by the local Granite checkpoint.

    Called once (lazily) when the HybridChunker is first needed.  The result
    is stored on the DocumentProcessor instance to avoid repeated loading.
    """
    from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
    from transformers import AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(
        "./granite_tokenizer",
        trust_remote_code=True,
        local_files_only=True,
    )

    class _LocalGraniteTokenizer(BaseTokenizer):
        max_tokens: int = 4096
        _hf_tokenizer: object = None

        def __init__(self, hf_tok):
            super().__init__(max_tokens=2048)
            self._hf_tokenizer = hf_tok

        def count_tokens(self, text: str) -> int:
            return len(self._hf_tokenizer.encode(text, add_special_tokens=False))

        def get_max_tokens(self) -> int:
            return self.max_tokens

        def get_tokenizer(self) -> object:
            return self._hf_tokenizer

        def __call__(self, text: str):
            return self._hf_tokenizer(text, return_tensors="pt", add_special_tokens=False)

        def encode(self, text: str) -> list[int]:
            return self._hf_tokenizer.encode(text, add_special_tokens=False)

    return _LocalGraniteTokenizer(hf_tokenizer)


class DocumentProcessor:
    """Process PDF documents into chunks for vector storage.

    Uses IBM's Docling library to convert PDFs and then applies
    Docling's HybridChunker directly on the DoclingDocument for
    structure-aware, tokenization-aware chunking.

    Hybrid indexing strategy:
        - **Hybrid chunks**: Document-structure-aware passages for retrieval.
        - **Summary chunk**: Paper-level overview for broad questions.

    Metadata per chunk:
        - paper: Display title (LLM-extracted when available, else filename).
        - domain: Broad domain category (e.g., "Machine Learning").
        - paper_chunk: Incremental ID ("chunk_001", "chunk_002", …) or "summary".
        - chunk_type: "content" | "summary".
        - source: Original filename.
        - file_path: Full path on disk.
        - chunk_index: 0-based position among content chunks.
        - total_chunks: Total number of content chunks.

    """

    def __init__(self):
        self._converter = None
        self._chunker = None

    @property
    def chunker(self):
        """Lazy-initialize the Docling HybridChunker."""
        if self._chunker is None:
            from docling.chunking import HybridChunker

            docling_tokenizer = _build_granite_tokenizer()
            self._chunker = HybridChunker(
                tokenizer=docling_tokenizer,
                merge_peers=True,
                max_tokens=8192,
            )
        return self._chunker

    @property
    def converter(self):
        """Lazy-initialize the Docling DocumentConverter."""
        if self._converter is None:
            from docling.document_converter import DocumentConverter
            self._converter = DocumentConverter()
        return self._converter


    @staticmethod
    def _paper_name_from_file(file_name: str) -> str:
        """Derive a human-readable paper name from the uploaded filename.

        Used only as a fallback when the LLM profile extraction does not
        yield a usable title.
        """
        name = os.path.splitext(file_name)[0]
        return re.sub(r"[-_]+", " ", name).strip().title()


    @staticmethod
    def generate_chunk_id(document_id: str, paper_chunk: str) -> str:
        """Derive a deterministic chunk ID from document_id and chunk label."""
        key = f"{document_id}:{paper_chunk}"
        return hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()

    def process(self, file_path: str, model: str = None, extra_metadata: dict = None,
                on_step: Optional[Callable[[str], None]] = None) -> dict:
        """Process a single PDF into semantically-chunked documents plus a summary.

        One LLM call produces the full :class:`PaperProfile` (title, domain,
        summary, concepts, methods, findings).  The profile is attached to
        the return value so the KG extractor can reuse it without making
        its own round-trip.

        Args:
            file_path: Path to the PDF file.
            model: Ollama model name for profile extraction.  If ``None``
                the LLM step is skipped and only filename-derived metadata
                is produced.
            extra_metadata: Optional dict of additional metadata fields to
                merge into every chunk (e.g. document_id, ingest_id).
            on_step: Optional callback ``(message: str)`` invoked before
                each major processing step so callers can report sub-file
                progress (e.g. update a Streamlit progress bar text).

        Returns:
            Dict with:
                - ``chunks``: list of LangChain :class:`Document` objects
                  (summary chunk first if available).
                - ``markdown``: raw Markdown text (for reuse by KG extractor).
                - ``paper_title``: display title used for chunk metadata.
                - ``domain``: classified domain category.
                - ``profile``: the full :class:`PaperProfile` or ``None``
                  when LLM extraction was skipped/failed.
        """
        file_name = os.path.basename(file_path)

        def _step(msg: str) -> None:
            """Emit *msg* to stdout and invoke the caller's on_step hook."""
            print(f"  [processor] {msg}", flush=True)
            if on_step:
                on_step(msg)

        _step(f"Converting PDF → document structure ({file_name})...")
        result = self.converter.convert(file_path)
        dl_doc = result.document
        markdown_text = dl_doc.export_to_markdown()
        print(f"  [processor] PDF converted — {len(markdown_text):,} chars of text", flush=True)

        try:
            txt_dir = os.path.join("data", "txt")
            os.makedirs(txt_dir, exist_ok=True)
            stem = os.path.splitext(file_name)[0]
            txt_path = os.path.join(txt_dir, f"{stem}.md")
            with open(txt_path, "w", encoding="utf-8") as fh:
                fh.write(markdown_text)
            print(f"  [processor] Markdown saved → {txt_path}", flush=True)
        except OSError as exc:
            logger.warning("Could not save markdown to %s: %s", txt_path, exc)

        profile: PaperProfile | None = None
        if model:
            _step(f"Extracting paper profile with LLM ({model})...")
            llm = get_llm(model=model, temperature=0.1, require_json=True)
            profile = extract_paper_profile(llm, markdown_text, file_name)
            title_preview = profile.title if profile else "n/a"
            print(f"  [processor] Profile extracted — title: '{title_preview}'", flush=True)

        fallback_title = self._paper_name_from_file(file_name)
        if profile and profile.title.strip():
            paper_title = profile.title.strip()
        else:
            paper_title = fallback_title
        domain = profile.domain if profile else "Other"
        summary_text = profile.summary if profile else ""

        _step("Chunking document (structure-aware)...")
        raw_chunks = list(self.chunker.chunk(dl_doc=dl_doc))
        print(f"  [processor] Created {len(raw_chunks)} chunks", flush=True)

        base_metadata = {
            "paper": paper_title,
            "domain": domain,
            "total_chunks": len(raw_chunks),
            "source": file_name,
            "file_path": file_path,
        }
        enriched: list[Document] = []
        for i, chunk in enumerate(raw_chunks):
            enriched_text = self.chunker.contextualize(chunk=chunk)
            enriched.append(Document(
                page_content=enriched_text,
                metadata={
                    **base_metadata,
                    "paper_chunk": f"chunk_{i + 1:03d}",
                    "chunk_type": "content",
                    "chunk_index": i,
                },
            ))

        if summary_text:
            summary_doc = Document(
                page_content=summary_text,
                metadata={
                    **base_metadata,
                    "paper_chunk": "summary",
                    "chunk_type": "summary",
                    "chunk_index": -1,
                },
            )
            enriched.insert(0, summary_doc)

        if extra_metadata:
            for doc in enriched:
                doc.metadata.update(extra_metadata)

        for doc in enriched:
            doc_id = doc.metadata.get("document_id", doc.metadata.get("source", file_name))
            chunk_label = doc.metadata.get("paper_chunk", "unknown")
            doc.metadata["chunk_id"] = self.generate_chunk_id(doc_id, chunk_label)

        return {
            "chunks": enriched,
            "markdown": markdown_text,
            "paper_title": paper_title,
            "domain": domain,
            "profile": profile,
        }
