"""
Local RAG Application using Ollama + Sentence Transformers.
No API keys required - runs entirely on your machine.
Enhanced with Excel processing and financial analysis capabilities.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from config import settings
from local_llm import LocalLLM, LocalEmbedder
from protocols import LLMProvider, EmbeddingProvider

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

load_dotenv()


class SimpleRAG:
    """Simple RAG implementation for local use with Excel and financial analysis support."""

    # Supported Excel extensions
    EXCEL_EXTENSIONS = {'.xlsx', '.xlsm', '.xls', '.csv', '.tsv'}

    def __init__(
        self,
        docs_folder: str = "./documents",
        llm_model: str = "llama3.2",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm: Optional[LLMProvider] = None,
        embedder: Optional[EmbeddingProvider] = None,
    ):
        """
        Initialize the RAG system.

        Args:
            docs_folder: Path to folder containing PDF/text/Excel documents
            llm_model: Ollama model name (ignored if *llm* is provided)
            embedding_model: Sentence transformer model name (ignored if *embedder* is provided)
            llm: Optional pre-built LLM provider (for dependency injection / testing)
            embedder: Optional pre-built embedding provider (for dependency injection / testing)
        """
        self.docs_folder = Path(docs_folder)
        self.docs_folder.mkdir(exist_ok=True)

        if llm is not None:
            self.llm = llm
        else:
            logger.info(f"Initializing LocalLLM with model: {llm_model}")
            self.llm = LocalLLM(model=llm_model)

        if embedder is not None:
            self.embedder = embedder
        else:
            logger.info(f"Initializing LocalEmbedder with model: {embedding_model}")
            self.embedder = LocalEmbedder(model_name=embedding_model)

        # Document store (simple in-memory)
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[list] = []

        # Excel processor and financial analyzer (lazy loaded)
        self._excel_processor = None
        self._charlie_analyzer = None

        # Embedding cache directory
        self._cache_dir = Path(settings.embedding_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Load documents on init
        self._load_documents()

    @property
    def excel_processor(self):
        """Lazy-load Excel processor."""
        if self._excel_processor is None:
            try:
                from excel_processor import ExcelProcessor
                self._excel_processor = ExcelProcessor(str(self.docs_folder))
            except ImportError:
                logger.warning("Excel processor not available. Install openpyxl and pandas.")
        return self._excel_processor

    @property
    def charlie_analyzer(self):
        """Lazy-load financial analyzer."""
        if self._charlie_analyzer is None:
            try:
                from financial_analyzer import CharlieAnalyzer
                self._charlie_analyzer = CharlieAnalyzer()
            except ImportError:
                logger.warning("Financial analyzer not available.")
        return self._charlie_analyzer

    # ------------------------------------------------------------------
    # Embedding cache helpers
    # ------------------------------------------------------------------

    def _embedding_cache_key(self, file_path: Path) -> str:
        """Generate a cache key from filename + modification time."""
        mtime = file_path.stat().st_mtime
        return hashlib.sha256(f"{file_path.name}:{mtime}".encode()).hexdigest()

    def _load_cached_embeddings(self, cache_key: str):
        """Load cached embeddings from disk via joblib, or return None."""
        cache_file = self._cache_dir / f"{cache_key}.joblib"
        if cache_file.exists():
            try:
                import joblib
                data = joblib.load(cache_file)
                logger.info(f"Loaded cached embeddings: {cache_key[:12]}...")
                return data  # (documents_list, embeddings_list)
            except Exception as e:
                logger.debug(f"Cache read failed: {e}")
        return None

    def _save_cached_embeddings(self, cache_key: str, documents: list, embeddings: list):
        """Save embeddings to disk cache."""
        cache_file = self._cache_dir / f"{cache_key}.joblib"
        try:
            import joblib
            joblib.dump((documents, embeddings), cache_file)
            logger.debug(f"Saved embeddings cache: {cache_key[:12]}...")
        except Exception as e:
            logger.debug(f"Cache write failed: {e}")

    # ------------------------------------------------------------------
    # Document loading
    # ------------------------------------------------------------------

    def _extract_text(self, file_path: Path) -> Optional[str]:
        """Extract text from a PDF or text file. Returns None on failure."""
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            import fitz
            doc = fitz.open(file_path)
            text = "".join(page.get_text() for page in doc)
            doc.close()
            return text if text.strip() else None

        if suffix in (".txt", ".md"):
            text = file_path.read_text(encoding="utf-8")
            return text if text.strip() else None

        return None

    def _load_documents(self):
        """Load and process documents from the docs folder (PDF, TXT, MD, Excel, CSV)."""
        logger.info(f"Loading documents from {self.docs_folder}")

        for file_path in self.docs_folder.glob("*"):
            try:
                suffix = file_path.suffix.lower()

                # Try embedding cache for this file
                cache_key = self._embedding_cache_key(file_path)
                cached = self._load_cached_embeddings(cache_key)
                if cached is not None:
                    docs, embs = cached
                    self.documents.extend(docs)
                    self.embeddings.extend(embs)
                    continue

                file_docs: List[Dict[str, Any]] = []

                if suffix in (".pdf", ".txt", ".md"):
                    text = self._extract_text(file_path)
                    if text:
                        doc_type = "pdf" if suffix == ".pdf" else "text"
                        chunks = self._chunk_text(
                            text,
                            chunk_size=settings.chunk_size,
                            overlap=settings.chunk_overlap,
                        )
                        for chunk in chunks:
                            file_docs.append({
                                "source": file_path.name,
                                "content": chunk,
                                "type": doc_type,
                            })
                        logger.info(f"Loaded {len(chunks)} chunks from {file_path.name}")

                elif suffix in self.EXCEL_EXTENSIONS:
                    excel_docs = self._process_excel_file(file_path)
                    file_docs.extend(excel_docs)
                    logger.info(f"Loaded {len(excel_docs)} chunks from Excel file {file_path.name}")

                # Generate and cache embeddings for this file's docs
                if file_docs:
                    texts = [d["content"] for d in file_docs]
                    embs = self.embedder.embed_batch(texts)
                    self._save_cached_embeddings(cache_key, file_docs, embs)
                    self.documents.extend(file_docs)
                    self.embeddings.extend(embs)

            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        if self.documents:
            logger.info(f"Total: {len(self.documents)} chunks indexed")
        else:
            logger.warning("No documents found in the documents folder")

    def _process_excel_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process Excel file into RAG-compatible document chunks."""
        if self.excel_processor is None:
            logger.warning(f"Excel processor not available, skipping {file_path.name}")
            return []

        try:
            from excel_processor import process_excel_for_rag
            return process_excel_for_rag(file_path, str(self.docs_folder))
        except Exception as e:
            logger.error(f"Failed to process Excel file {file_path}: {e}")
            return []

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks if chunks else [text]

    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        """Calculate cosine similarity between two vectors."""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """
        Retrieve the most relevant documents for a query.

        Args:
            query: User's question
            top_k: Number of documents to retrieve

        Returns:
            List of relevant document chunks
        """
        if not self.documents:
            return []

        # Clamp top_k to valid range
        top_k = max(1, min(top_k, settings.max_top_k, len(self.documents)))

        # Embed the query
        query_embedding = self.embedder.embed(query)

        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, sim))

        # Sort by similarity and get top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[:top_k]]

        return [self.documents[i] for i in top_indices]

    def answer(self, query: str) -> str:
        """
        Answer a question using RAG with financial analysis support.

        Args:
            query: User's question

        Returns:
            Generated answer
        """
        # Enforce query length limit
        if len(query) > settings.max_query_length:
            return f"Query too long. Maximum length is {settings.max_query_length} characters."

        # Retrieve relevant documents
        relevant_docs = self.retrieve(query, top_k=settings.top_k)

        if not relevant_docs:
            return "No documents loaded. Please add PDF, text, or Excel files to the 'documents' folder."

        # Check if query involves financial analysis
        is_financial = self._is_financial_query(query)

        # Build context from retrieved documents
        context_parts = []
        excel_data = []

        for doc in relevant_docs:
            source_info = f"[Source: {doc['source']}]"
            doc_type = doc.get('type', 'unknown')

            if doc_type == 'excel':
                financial_type = doc.get('metadata', {}).get('financial_type', '')
                if financial_type:
                    source_info += f" [Type: {financial_type}]"
                excel_data.append(doc)

            context_parts.append(f"{source_info}\n{doc['content']}")

        context = "\n\n".join(context_parts)

        # Enhanced prompt for financial queries
        if is_financial and self.charlie_analyzer and excel_data:
            prompt = self._build_financial_prompt(query, context, excel_data)
        else:
            # Standard RAG prompt
            prompt = f"""You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context to answer. If the answer is not in the context, say so.

Context:
{context}

Question: {query}

Answer:"""

        # Generate answer
        try:
            answer = self.llm.generate(prompt)
            return answer.strip()
        except Exception as e:
            return f"Error generating answer: {e}"

    def _is_financial_query(self, query: str) -> bool:
        """Check if query involves financial analysis."""
        financial_keywords = [
            'ratio', 'margin', 'profit', 'revenue', 'income', 'expense',
            'cash flow', 'budget', 'variance', 'roe', 'roa', 'roi',
            'liquidity', 'leverage', 'debt', 'equity', 'asset', 'liability',
            'growth', 'trend', 'forecast', 'analysis', 'financial',
            'balance sheet', 'income statement', 'p&l', 'cfo', 'ebitda'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in financial_keywords)

    def _build_financial_prompt(self, query: str, context: str, excel_data: List[Dict]) -> str:
        """Build an enhanced prompt for financial queries."""
        analysis_text = ""

        if self.charlie_analyzer:
            try:
                for doc in excel_data[:1]:
                    table_struct = doc.get('metadata', {}).get('table_structure', {})
                    if table_struct:
                        analysis_text = f"\nDetected data structure: {table_struct.get('type', 'financial data')}"
            except Exception as e:
                logger.debug(f"Could not run financial analysis: {e}")

        return f"""You are a senior financial analyst with CFO-level expertise, inspired by Charlie Munger's
analytical framework. Focus on fundamentals, cash flow, and sustainable competitive advantages.

USER QUERY: {query}

FINANCIAL DATA CONTEXT:
{context}
{analysis_text}

ANALYSIS FRAMEWORK (Charlie Munger approach):
1. What are the key value drivers?
2. What could go wrong? (Inversion thinking)
3. Is there a margin of safety?
4. Focus on cash flows, not just accounting profits
5. Look for sustainable competitive advantages

Provide a comprehensive, actionable analysis addressing the query.
Include specific numbers from the data and cite your sources.
If performing calculations, show your work.

Answer:"""

    def reload_documents(self):
        """Reload documents from the folder."""
        self.documents = []
        self.embeddings = []
        self._load_documents()


# CLI-only entry point (not used by Streamlit - see streamlit_app_local.py)
if __name__ == "__main__":
    rag = SimpleRAG(
        docs_folder="./documents",
        llm_model=os.getenv("OLLAMA_MODEL", settings.llm_model),
        embedding_model=os.getenv("EMBEDDING_MODEL", settings.embedding_model),
    )

    print("\n" + "=" * 50)
    print("Local RAG System Ready!")
    print("=" * 50)
    print(f"Documents loaded: {len(rag.documents)} chunks")
    print(f"Documents folder: {rag.docs_folder.absolute()}")
    print("\nAdd PDF or TXT files to the 'documents' folder to get started.")
    print("=" * 50 + "\n")

    while True:
        query = input("\nAsk a question (or 'quit' to exit): ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        if query:
            print("\nSearching and generating answer...")
            answer = rag.answer(query)
            print(f"\nAnswer: {answer}")
