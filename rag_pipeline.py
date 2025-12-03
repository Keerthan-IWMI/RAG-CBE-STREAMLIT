import os
import fitz
import re
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
import logging
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Add these imports at top of your file (if not already present)
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Tuple, Optional, Dict
import numpy as np
import heapq
import math

class RelevanceChecker:
    """
    RelevanceChecker: rerank, threshold, optional contextual compression.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        cross_encoder_name: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        threshold: float = 0.70,
        min_docs: int = 2,
        max_docs: int = 6,
        enable_compression: bool = True,
        compression_top_sentences: int = 3,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        fallback_to_cosine: bool = True
    ):
        self.embedding_model = embedding_model
        self.cross_encoder = None
        self.cross_encoder_name = cross_encoder_name

        if cross_encoder_name:
            try:
                self.cross_encoder = CrossEncoder(cross_encoder_name)
                logger.info(f"Loaded CrossEncoder: {cross_encoder_name}")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder '{cross_encoder_name}': {e}")
                self.cross_encoder = None

        self.threshold = threshold
        self.min_docs = min_docs
        self.max_docs = max_docs
        self.enable_compression = enable_compression
        self.compression_top_sentences = compression_top_sentences
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.fallback_to_cosine = fallback_to_cosine

    # -------------------------
    # Public API
    # -------------------------
    def filter_documents(
        self,
        question: str,
        docs: List,
        doc_embeddings: Optional[np.ndarray] = None
    ) -> List[Tuple[object, float]]:

        logger.info(f"Filtering {len(docs)} retrieved chunks for question: {question}")

        if not docs:
            logger.warning("No documents retrieved.")
            return []

        # 1) Scoring
        if self.cross_encoder is not None:
            scored = self._score_with_crossencoder(question, docs)
        else:
            scored = self._score_with_cosine(question, docs, doc_embeddings)

        # Log all scored chunks
        for i, (doc, score) in enumerate(scored):
            logger.info(f"[Score] Rank={i+1} Score={score:.4f} Content={doc.page_content[:150]}...")

        # 2) Thresholding
        filtered = [(d, s) for d, s in scored if s >= self.threshold]
        logger.info(f"Chunks above threshold {self.threshold}: {len(filtered)}")

        if len(filtered) < self.min_docs:
            logger.info(f"Below min_docs={self.min_docs}, selecting top {self.min_docs} anyway.")
            filtered = scored[: self.min_docs]

        filtered = filtered[: self.max_docs]

        # Log filtered results
        for doc, score in filtered:
            logger.info(f"[Selected] Score={score:.4f} Content={doc.page_content[:150]}...")

        # 3) Compression
        if self.enable_compression:
            compressed = []
            for doc, score in filtered:
                logger.info(f"Compressing chunk (score={score:.4f}): {doc.page_content[:150]}...")
                compressed_doc = self._compress_document(question, doc, top_k=self.compression_top_sentences)
                logger.info(f"[Compressed Result] {compressed_doc.page_content[:200]}...")
                compressed.append((compressed_doc, score))
            filtered = compressed

        return filtered

    # -------------------------
    # Internal scoring helpers
    # -------------------------
    def _score_with_crossencoder(self, question: str, docs: List) -> List[Tuple[object, float]]:
        # Format as tuples (query, passage)
        cross_input = [(question, doc.page_content) for doc in docs]
        try:
            scores = self.cross_encoder.predict(cross_input, batch_size=self.batch_size)
            logger.info("Cross-encoder scoring successful.")
        except Exception as e:
            logger.warning(f"Cross-encoder failed: {e}, falling back to cosine.")
            return self._score_with_cosine(question, docs)

        scores = self._minmax_normalize(np.array(scores))
        return sorted(list(zip(docs, scores.tolist())), key=lambda x: x[1], reverse=True)


    def _score_with_cosine(self, question: str, docs: List, doc_embeddings: Optional[np.ndarray] = None):
        logger.info("Using cosine similarity scoring...")

        q_emb = self.embedding_model.encode([question], show_progress_bar=False, convert_to_numpy=True)
        if self.normalize_embeddings:
            q_emb = self._l2_normalize(q_emb)

        if doc_embeddings is None:
            texts = [d.page_content for d in docs]
            doc_embeddings = self.embedding_model.encode(texts, batch_size=self.batch_size,
                                                         show_progress_bar=False, convert_to_numpy=True)
        if self.normalize_embeddings:
            doc_embeddings = self._l2_normalize(doc_embeddings)

        sims = np.dot(doc_embeddings, q_emb.T).reshape(-1)
        sims = (sims + 1.0) / 2.0  # normalize to 0–1

        return sorted(list(zip(docs, sims.tolist())), key=lambda x: x[1], reverse=True)

    # -------------------------
    # Contextual compression
    # -------------------------
    def _compress_document(self, question: str, doc, top_k: int = 3):
        text = doc.page_content
        sentences = self._split_sentences(text)

        logger.info(f"Splitting into {len(sentences)} sentences for compression.")

        if not sentences:
            return doc

        q_emb = self.embedding_model.encode([question], show_progress_bar=False, convert_to_numpy=True)
        sent_embs = self.embedding_model.encode(sentences, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)

        if self.normalize_embeddings:
            q_emb = self._l2_normalize(q_emb)
            sent_embs = self._l2_normalize(sent_embs)

        sims = np.dot(sent_embs, q_emb.T).reshape(-1)

        # Log top sentences
        idx_scores = list(enumerate(sims))
        idx_scores_sorted = sorted(idx_scores, key=lambda x: x[1], reverse=True)
        for i, (idx, score) in enumerate(idx_scores_sorted[:top_k]):
            logger.info(f"[Compression Sentence #{i+1}] Score={score:.4f} Sentence={sentences[idx][:200]}")

        top_idx = [i for i, _ in idx_scores_sorted[:top_k]]
        top_idx.sort()

        compressed_text = " ".join([sentences[i] for i in top_idx]).strip()

        compressed_doc = type(doc)(
            page_content=compressed_text,
            metadata={**getattr(doc, "metadata", {})}
        )
        return compressed_doc

    # -------------------------
    # Utilities
    # -------------------------
    @staticmethod
    def _minmax_normalize(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-12)

    @staticmethod
    def _l2_normalize(x):
        denom = np.linalg.norm(x, axis=1, keepdims=True)
        denom[denom == 0] = 1e-12
        return x / denom

    @staticmethod
    def _split_sentences(text: str):
        parts = re.split(r'(?<=[\.\?\!])\s+', text)
        return [p.strip() for p in parts if p.strip()]
class ConversationManager:
    """Manages conversation history with model-aware token limits"""
    
    # Context window limits (tokens)
    MODEL_LIMITS = {
        "deepseek": 64000,    # DeepSeek-V3
        "azure": 128000,      # GPT-4o-mini
    }
    
    def __init__(self, llm_type: str, reserve_tokens: int = 8000):
        """
        Args:
            llm_type: "azure" or "deepseek"
            reserve_tokens: Tokens to reserve for system prompt + retrieved docs + response
        """
        self.llm_type = llm_type.lower()
        self.max_context = self.MODEL_LIMITS.get(self.llm_type, 64000)
        self.reserve_tokens = reserve_tokens
        self.available_for_history = self.max_context - self.reserve_tokens
        
        # Initialize tokenizer for counting
        try:
            # GPT-4o and DeepSeek use similar tokenization
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            logger.warning("Failed to load tiktoken, using fallback estimation")
            self.tokenizer = None
        
        self.history = []  # List of {"role": "user/assistant", "content": "..."}
        
        logger.info(f"ConversationManager initialized: {self.llm_type}, "
                   f"max={self.max_context}, available_for_history={self.available_for_history}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return len(text) // 4
    
    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in message list"""
        total = 0
        for msg in messages:
            # Count content tokens
            total += self.count_tokens(msg["content"])
            # Add overhead for message formatting (~4 tokens per message)
            total += 4
        return total
    
    def add_exchange(self, user_message: str, assistant_message: str):
        """Add a Q&A pair to history with automatic truncation"""
        # Add new messages
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": assistant_message})
        
        # Truncate if needed
        self._truncate_to_fit()
    
    def _truncate_to_fit(self):
        """Remove oldest messages until history fits in available token budget"""
        current_tokens = self.count_messages_tokens(self.history)
        
        # Keep removing oldest Q&A pairs until we fit
        while current_tokens > self.available_for_history and len(self.history) > 2:
            # Remove oldest Q&A pair (first 2 messages)
            removed = self.history[:2]
            self.history = self.history[2:]
            
            removed_tokens = self.count_messages_tokens(removed)
            current_tokens -= removed_tokens
            
            logger.info(f"Truncated conversation: removed {removed_tokens} tokens, "
                       f"remaining={current_tokens}/{self.available_for_history}")
        
        # Log if we're getting close to limit
        if current_tokens > self.available_for_history * 0.8:
            pairs = len(self.history) // 2
            logger.warning(f"Conversation history at 80% capacity: {current_tokens} tokens, {pairs} pairs")
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get current conversation history"""
        return self.history.copy()
    
    def get_history_tokens(self) -> int:
        """Get current token count of history"""
        return self.count_messages_tokens(self.history)
    
    def clear(self):
        """Clear conversation history"""
        self.history = []
        logger.info("Conversation history cleared")
    
    def get_stats(self) -> Dict:
        """Get conversation statistics"""
        pairs = len(self.history) // 2
        tokens = self.count_messages_tokens(self.history)
        
        return {
            "total_exchanges": pairs,
            "history_tokens": tokens,
            "available_tokens": self.available_for_history,
            "utilization_percent": round((tokens / self.available_for_history) * 100, 1),
            "model": self.llm_type,
            "max_context": self.max_context
        }
#sentence-transformers/all-mpnet-base-v2
class PDFExtractor:
    """Handles PDF extraction with layout preservation"""
    
    def __init__(self):
        self.header_footer_margin = 50
        self.min_text_length = 50
        self.heading_font_threshold = 13
    
    def extract_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract content from PDF with structure preservation
        Returns: List of content blocks with metadata
        """
        try:
            content_blocks = self._extract_with_layout(pdf_path)
            
            if not content_blocks:
                logger.warning(f"Layout extraction failed, using fallback for {pdf_path}")
                content_blocks = self._fallback_extraction(pdf_path)
            
            # Merge small adjacent blocks
            content_blocks = self._merge_blocks(content_blocks)
            
            return content_blocks
        
        except Exception as e:
            logger.error(f"PDF extraction failed for {pdf_path}: {e}")
            return []
    
    def _extract_with_layout(self, pdf_path: str) -> List[Dict]:
        """Extract text with layout and structure preservation"""
        doc = fitz.open(pdf_path)
        all_content = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_height = page.rect.height
            
            # Get text blocks with layout info
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                
                bbox = block["bbox"]
                
                # Skip headers and footers
                if (bbox[1] < self.header_footer_margin or 
                    bbox[3] > page_height - self.header_footer_margin):
                    continue
                
                # Extract text and font information
                text_lines = []
                font_sizes = []
                
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                        font_sizes.append(span["size"])
                    
                    if line_text.strip():
                        text_lines.append(line_text.strip())
                
                if not text_lines:
                    continue
                
                text = " ".join(text_lines)
                
                if len(text.strip()) < self.min_text_length:
                    continue
                
                # Detect headings by font size
                avg_font_size = np.mean(font_sizes) if font_sizes else 11
                content_type = "heading" if avg_font_size > self.heading_font_threshold else "paragraph"
                
                all_content.append({
                    "text": text,
                    "page": page_num + 1,
                    "type": content_type,
                    "bbox": bbox
                })
            
            # Extract tables separately
            tables = self._extract_tables(page, page_num + 1)
            all_content.extend(tables)
        
        doc.close()
        return all_content
    
    def _extract_tables(self, page, page_num: int) -> List[Dict]:
        """Extract tables from page using PyMuPDF"""
        tables = []
        
        try:
            tabs = page.find_tables()
            
            for i, table in enumerate(tabs):
                df = table.to_pandas()
                
                if df.empty:
                    continue
                
                table_text = f"Table {i+1}:\n{df.to_string(index=False)}"
                
                tables.append({
                    "text": table_text,
                    "page": page_num,
                    "type": "table",
                    "bbox": table.bbox
                })
        
        except Exception as e:
            logger.debug(f"Table extraction failed on page {page_num}: {e}")
        
        return tables
    
    def _fallback_extraction(self, pdf_path: str) -> List[Dict]:
        """Simple fallback if advanced extraction fails"""
        doc = fitz.open(pdf_path)
        content = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            
            if text.strip():
                content.append({
                    "text": text.strip(),
                    "page": page_num + 1,
                    "type": "text"
                })
        
        doc.close()
        return content
    
    def _merge_blocks(self, content_blocks: List[Dict]) -> List[Dict]:
        """Merge small adjacent blocks on same page"""
        if not content_blocks:
            return []
        
        merged = []
        current_block = None
        
        for block in content_blocks:
            # Always keep tables separate
            if block["type"] == "table":
                if current_block:
                    merged.append(current_block)
                    current_block = None
                merged.append(block)
                continue
            
            if current_block is None:
                current_block = block.copy()
                continue
            
            # Merge if same page and combined text not too long
            if (block["page"] == current_block["page"] and 
                len(current_block["text"]) + len(block["text"]) < 800):
                current_block["text"] += " " + block["text"]
            else:
                merged.append(current_block)
                current_block = block.copy()
        
        if current_block:
            merged.append(current_block)
        
        return merged
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize extracted text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\b\d{1,3}\s*$', '', text)
        text = re.sub(r'[^\w\s\.\,\:\;\-\(\)\[\]\"\'\%\$\#\@\!\?\/\&\+\=\*]', '', text)
        text = re.sub(r'\s+([.,;:!?)])', r'\1', text)
        text = re.sub(r'([(])\s+', r'\1', text)
        return text.strip()


class RAGPipeline:
    """Main RAG pipeline for PDF question answering"""
    def __init__(
        self,
        pdf_folder: str,
        index_file: str,
        model_params: dict,
        reserve_tokens: int = 8000,
    ):
        # Paths
        self.pdf_folder = pdf_folder
        self.index_file = index_file
        
        # Detect LLM type
        llm_type = model_params.get("llm_type", "").lower()
        if llm_type not in ("azure", "deepseek"):
            raise ValueError("model_params['llm_type'] must be 'azure' or 'deepseek'")
        
        self.llm_type = llm_type
        
        # Initialize conversation manager
        self.conversation_manager = ConversationManager(
            llm_type=self.llm_type,
            reserve_tokens=reserve_tokens
        )
        
        # ------------------------------------------------------------------
        # 3. Azure OpenAI
        # ------------------------------------------------------------------
        if self.llm_type == "azure":
            required = ["azure_key", "azure_endpoint", "azure_deployment"]
            missing = [k for k in required if k not in model_params]
            if missing:
                raise ValueError(f"Missing Azure keys in model_params: {missing}")

            self.llm_client = AzureOpenAI(
                api_key=model_params["azure_key"],
                api_version="2024-02-15-preview",
                azure_endpoint=model_params["azure_endpoint"],
            )
            self.azure_deployment = model_params["azure_deployment"]

        # ------------------------------------------------------------------
        # 4. DeepSeek (HF Inference API)
        # ------------------------------------------------------------------
        else:  # deepseek
            if "hf_token" not in model_params:
                raise ValueError("hf_token is required for DeepSeek")
            self.hf_token      = model_params["hf_token"]
            self.deepseek_url  = model_params.get("deepseek_url",
                                "https://api.deepseek.com/v1/chat/completions")
            self.deepseek_model = model_params.get("deepseek_model", "deepseek-ai/DeepSeek-V3.1:novita")
        
        # Models
        self.embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        self.embedding_model.eval()
        # create relevance checker
        self.relevance_checker = RelevanceChecker(
            embedding_model=self.embedding_model,
            cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2",  # best-effort default
            threshold=0.65,
            min_docs=2,
            max_docs=6,
            enable_compression=False,
            compression_top_sentences=3
        )
        
        self.pdf_extractor = PDFExtractor()
        
        # Text splitter
        # self.text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=700,
        #     chunk_overlap=150,
        #     length_function=len,
        #     separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        #     is_separator_regex=False,
        # )

        #Simple sliding window chunker.
        # from langchain_text_splitters import CharacterTextSplitter

        # self.text_splitter = CharacterTextSplitter(
        #     chunk_size=512,
        #     chunk_overlap=50,
        #     separator="\n\n",   
        # )

        #Simple fixed length chunker.
        # from langchain_text_splitters import CharacterTextSplitter

        # self.text_splitter = CharacterTextSplitter(
        #     chunk_size=512,
        #     chunk_overlap=0,
        #     separator="\n\n",   
        # )

        #Semantic-ish chunker.
        self.text_splitter = SemanticChunker(
            embedding_model=self.embedding_model,
            base_chunk_size=512,
            base_overlap=50,
            sim_threshold=0.85,
        )
        """The Semantica actually worked better than the sliding window chunker."""
        """Fixed length is shape. But it migth not be good to pick up on context."""
        """Sliding window chunker actually made it better. I'll have to tweak more with the chunk_size and chunk_overlap."""
        """When I checked the chunks found in recursive splitter it did have the required texts, but the recursive splitter
           seems to split them at random points leading to a losss of context."""
        
        # Storage
        self.documents = []
        self.embeddings = None
        self.faiss_retriever = None
        self.bm25_retriever = None
        self.hybrid_retriever = None
    
    def build_index(self, progress_callback=None, status_callback=None):
        """Build index from PDFs in folder"""
        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.lower().endswith(".pdf")]
        
        if not pdf_files:
            raise ValueError("No PDF files found in folder")
        
        all_documents = []
        
        for i, pdf_file in enumerate(pdf_files):
            if status_callback:
                status_callback(f"Processing: {pdf_file} ({i+1}/{len(pdf_files)})")
            
            pdf_path = os.path.join(self.pdf_folder, pdf_file)
            
            try:
                # Extract content blocks
                content_blocks = self.pdf_extractor.extract_pdf(pdf_path)
                
                # Create smart chunks
                documents = self._create_chunks(content_blocks, pdf_file)
                all_documents.extend(documents)
                
                logger.info(f"Extracted {len(documents)} chunks from {pdf_file}")
            
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
                continue
            
            if progress_callback:
                progress_callback((i + 1) / len(pdf_files))
        
        if not all_documents:
            raise ValueError("No content extracted from PDFs")
        
        if status_callback:
            status_callback(f"Encoding {len(all_documents)} chunks...")
        
        # Encode documents
        texts = [doc.page_content for doc in all_documents]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Build retrievers
        self._build_retrievers(all_documents, texts, embeddings)
        
        # Save index
        self.documents = all_documents
        self.embeddings = embeddings
        self._save_index()
        
        if status_callback:
            status_callback(f"✅ Indexed {len(all_documents)} chunks from {len(pdf_files)} PDFs")
        
        return len(all_documents)
    
    def _create_chunks(self, content_blocks: List[Dict], pdf_name: str) -> List[Document]:
        """Create smart chunks from content blocks"""
        documents = []
        
        for block in content_blocks:
            text = self.pdf_extractor.clean_text(block["text"])
            
            if len(text) < 50:
                continue
            
            # Keep tables and short content as single chunks
            if block["type"] == "table" or len(text) < 600:
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": pdf_name,
                        "page": block["page"],
                        "type": block["type"]
                    }
                ))
            else:
                # Split long content
                chunks = self.text_splitter.split_text(text)
                
                for i, chunk in enumerate(chunks):
                    documents.append(Document(
                        page_content=chunk,
                        metadata={
                            "source": pdf_name,
                            "page": block["page"],
                            "type": block["type"],
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        }
                    ))
        
        return documents
    
    def _build_retrievers(self, documents: List[Document], texts: List[str], embeddings: np.ndarray):
        """Build FAISS and BM25 retrievers"""
        # FAISS retriever
        faiss_index = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            embedding=lambda x: self.embedding_model.encode(x, normalize_embeddings=True),
            metadatas=[doc.metadata for doc in documents]
        )
        self.faiss_retriever = faiss_index.as_retriever(search_kwargs={"k": 20})
        
        # BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 20
        
        # Hybrid retriever
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[self.faiss_retriever, self.bm25_retriever],
            weights=[0.85, 0.15]
        )
    
    def _save_index(self):
        """Save index to disk"""
        with open(self.index_file, "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "embeddings": self.embeddings,
                "model": "sentence-transformers/all-mpnet-base-v2"
            }, f)
    
    def load_index(self):
        """Load existing index from disk"""
        if not os.path.exists(self.index_file):
            return False
        
        try:
            with open(self.index_file, "rb") as f:
                data = pickle.load(f)
            
            self.documents = data["documents"]
            self.embeddings = data["embeddings"]
            
            # Rebuild retrievers
            texts = [doc.page_content for doc in self.documents]
            self._build_retrievers(self.documents, texts, self.embeddings)
            
            logger.info(f"Loaded index with {len(self.documents)} chunks")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def query(self, question: str, top_k: int = 8) -> Tuple[str, List[Document]]:

        if self.hybrid_retriever is None:
            raise ValueError("Index not loaded. Call load_index() or build_index() first.")
        
        # Retrieve documents
        retrieved_docs = self.hybrid_retriever.invoke(question)
        
        if not retrieved_docs:
            answer = "I couldn't find relevant information in the documents."
            return answer, []
        
        top_docs = retrieved_docs[:top_k]
        filtered = self.relevance_checker.filter_documents(question, top_docs)
        # filtered is List[(Document, score)]
        filtered_docs = [d for d, s in filtered]
        # Generate answer with history
        answer = self._generate_answer_with_history(question, filtered_docs)
        
        # Store this exchange in history
        self.conversation_manager.add_exchange(question, answer)
        
        # Log conversation stats
        stats = self.conversation_manager.get_stats()
        logger.info(f"Conversation: {stats['total_exchanges']} exchanges, "
                f"{stats['history_tokens']} tokens ({stats['utilization_percent']}% of available)")
        
        return answer, filtered_docs
    
    def _generate_answer_with_history(self, question: str, context_docs: List[Document]) -> str:
        """Generate answer using LLM"""
        # Build context
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            src = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            doc_type = doc.metadata.get("type", "text")
            
            type_label = f"[{doc_type.upper()}] " if doc_type != "paragraph" else ""
            context_parts.append(
                f"[Source {i}: {src}, Page {page}] {type_label}\n{doc.page_content}\n"
            )

        context = "\n".join(context_parts)
        
        # DEBUG: log context that will be sent to LLM
        logger.info("======== FINAL CONTEXT SENT TO LLM ========")
        logger.info(context[:4000])  # truncate to avoid huge logs
        logger.info("===========================================")
        
        system_prompt = """You are the Circular Bioeconomy Decision Support Assistant (CBE-DSA) - an AI-powered chatbot developed to disseminate applied research and evidence-based insights from the International Water Management Institute (IWMI) and related partners.
Your primary goal is to help users, including policymakers, industry professionals, entrepreneurs, investors, and development partners, make informed, evidence-based decisions in the circular bioeconomy and sustainable waste management.
Role and Behaviour:
- Serve as a research-driven knowledge advisor, interpreting academic and technical content into concise, practical, and actionable insights.
- Remain accurate, context-aware, and user-oriented, tailoring responses to the user’s role (e.g., policymaker vs. entrepreneur).
- Use information derived from the uploaded IWMI documents and supplement your responses with logically linking information derived from the top 5 most matched documents
- In case the query falls completely outside the scope of the available materials, then clearly indicate that the requested information cannot be found otherwise try to answers by logically linking information in different documents and deducing potential response
- Maintain a logical and structured conversation flow, referring to previous exchanges when needed.
- For follow-up questions like "tell me more" or "explain that further", use the conversation history to provide coherent, contextual responses.
- Sense the tone of the question to understand if the user needs generic or specific information. In case of specific information needs do not dilute your response by too much supporting extra information. As a follow up ask the user if they require “any further information” to “support” the information provided. Responses in case of specific queries need to primarily retrieved from documents with maximum “phrase matches” from the question rather than maximum “keyword” matches.
-In case of generic questions, keep the response as a deduced conclusion from the top documents with maximum “keyword” matches and keep asking follow-up questions from the user to user to keep refining the responses until the user is satisfied.
Tone and Communication Style:
- Professional, clear, neutral, and factual.
- Use plain language; cite sources when needed.
- Emphasize practical impact and innovation.
Your responses should be rooted in IWMI’s body of knowledge and structured around practical decision-making needs, such as identifying optimal technologies, evaluating business viability, assessing policy implications, and identifying best-case scenarios.
Restrictions and Limitations:
- Do not fabricate references, data, or methodologies; however you can create linkages between different documents to deduce responses. In case of deduced responses, indicate to the user that it is a response deduced from your analysis.
- Always clarify if a recommendation is derived from evidence (explicitly present in documents) or an inferred interpretation (logical synthesis based on available content).
- Avoid expressing political bias, or speculative prejudices
- Refrain from giving prescriptive financial or legal advice.
Response Format:
- **Overview:** Provide a summary of the findings (20-30 words for specific questions, 50-100 words for response to generic questions).
- **Key Points:**
    • Use bullet points starting with “•”.
    • After EACH bullet point, insert a blank line (exactly one empty line).
    • Each bullet point must be one sentence only.
- **Implications:** Present the practical or policy relevance as bullet points (20-30 words for specific questions, 50-100 words for response to generic questions).
- **If comparative or quantitative data are available**, display them using a **Markdown table** (| Column | Column |).
- **Always cite sources** in [Source X] format after each claim and hyperlink the sources.
- **Avoid long paragraphs**; favour bullet points and tabular summaries for clarity.
- **Combine multiple document sources when possible and most logical** to provide integrated insights.
 
If there seems to be a better structure to the response then follow that, rather than the overview, key points and implications format.
 
CONVERSATION FLOW & ENGAGEMENT (MANDATORY)
- **Every response must end with a proactive, context-aware follow-up question or suggestion.**
- Prefer follow-up questions that can be answered using details from previous exchanges (history) or the current information (retrieved documents) to deepen the discussion.
- If no specific follow-up question fits based on history or current info, encourage continuation with open-ended prompts like "Do you have any further questions?" or "What else would you like to know about this topic?"
- This keeps the conversation alive and guides the user toward deeper insights.
\n
Information Use:
- For general/greeting questions, ignore retrieved content and respond from general knowledge.
- For research-specific queries, rely strictly on IWMI sources.
Your mission is to provide support to science-based decision-making and accelerate the adoption of optimized circular bioeconomy business models, guided by IWMI’s validated research and expertise."""
        
        user_prompt = f"""CONTEXT FROM DOCUMENTS:
{context}

QUESTION: {question}
\n
INSTRUCTIONS FOR ANSWERING:
- Use the above context **and** the full conversation history.
- Always prioritise the **most relevant** and **highest-matching** information.
- Follow the Response Format, the System Prompt, and all other detailed instructions strictly.
\n
ANSWER (with citations):"""
        
        # Build messages with conversation history
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        history = self.conversation_manager.get_history()
        if history:
            messages.extend(history)
            logger.debug(f"Including {len(history)} history messages "
                        f"({self.conversation_manager.get_history_tokens()} tokens)")

        # Add current query with context
        messages.append({"role": "user", "content": user_prompt})

        # Log total message tokens
        if self.conversation_manager.tokenizer:
            total_input_tokens = self.conversation_manager.count_messages_tokens(messages)
            logger.info(f"Total input tokens: {total_input_tokens} "
                    f"(limit: {self.conversation_manager.max_context})")
        
     # --------------------------------------------------------------
        # CALL THE SELECTED LLM
        # --------------------------------------------------------------
        try:
            if self.llm_type == "azure":
                resp = self.llm_client.chat.completions.create(
                    model=self.azure_deployment,
                    messages=messages,
                    max_tokens=3500,
                    temperature=0.1,
                )
                return resp.choices[0].message.content.strip()

            else:  # deepseek
                headers = {
                    "Authorization": f"Bearer {self.hf_token}",
                    "Content-Type":  "application/json"
                }
                payload = {
                    "model":       self.deepseek_model,
                    "messages":    messages,
                    "max_tokens":  3500,
                    "temperature": 0.1
                }

                logger.info(f"Calling DeepSeek API ({len(messages)} msgs)")
                r = requests.post(self.deepseek_url, headers=headers, json=payload, timeout=60)

                if r.status_code == 200:
                    data = r.json()
                    if data.get("choices"):
                        return data["choices"][0]["message"]["content"].strip()
                    else:
                        logger.error(f"DeepSeek unexpected payload: {data}")
                        return "Error: Unexpected DeepSeek response"
                else:
                    err = f"DeepSeek API {r.status_code}: {r.text}"
                    logger.error(err)
                    return f"Sorry, model error: {err}"

        except Exception as e:
            logger.error(f"LLM failure ({self.llm_type}): {e}")
            return "Sorry, I encountered an error generating the response."
    def clear_conversation(self):
        self.conversation_manager.clear()

    def get_conversation_stats(self) -> Dict:
        return self.conversation_manager.get_stats()
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        if not self.documents:
            return {"total_chunks": 0}
        
        stats = {
            "total_chunks": len(self.documents),
            "content_types": {}
        }
        
        for doc in self.documents:
            doc_type = doc.metadata.get("type", "unknown")
            stats["content_types"][doc_type] = stats["content_types"].get(doc_type, 0) + 1
        
        return stats

    ##Chunk checker. Only for debugging purposes.
    def debug_print_chunks_for_source(self, source_name: str, max_chunks: int = 20):
        """Print all (or first N) chunks for a given PDF source."""
        matched = [d for d in self.documents if d.metadata.get("source") == source_name]
        print(f"[DEBUG] Found {len(matched)} chunks for source='{source_name}'")
        for i, doc in enumerate(matched[:max_chunks], 1):
            print(f"\n--- Chunk {i} ---")
            print("metadata:", doc.metadata)
            preview = doc.page_content[:800].replace("\n", " ")
            print("text    :", preview, "...")

class SemanticChunker:
    """
    Very simple semantic-ish chunker:
    1) First split into small fixed chunks.
    2) Then merge adjacent chunks whose embeddings are very similar.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        base_chunk_size: int = 512,
        base_overlap: int = 50,
        sim_threshold: float = 0.85,
    ):
        from langchain_text_splitters import CharacterTextSplitter

        self.embedding_model = embedding_model
        self.base_splitter = CharacterTextSplitter(
            chunk_size=base_chunk_size,
            chunk_overlap=base_overlap,
            separator="\n\n",
        )
        self.sim_threshold = sim_threshold

    def split_text(self, text: str) -> list[str]:
        # 1) base split
        mini_chunks = self.base_splitter.split_text(text)
        if len(mini_chunks) <= 1:
            return mini_chunks

        # 2) embed all mini-chunks
        embs = self.embedding_model.encode(
            mini_chunks, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
        )

        # 3) merge adjacent chunks with high cosine similarity
        merged_chunks: list[str] = []
        current = mini_chunks[0]
        current_emb = embs[0]

        for i in range(1, len(mini_chunks)):
            sim = float(np.dot(current_emb, embs[i]))
            if sim >= self.sim_threshold:
                # same topic: merge
                current = current + " " + mini_chunks[i]
                # update embedding as average of both (approx)
                current_emb = (current_emb + embs[i]) / 2.0
            else:
                merged_chunks.append(current)
                current = mini_chunks[i]
                current_emb = embs[i]

        merged_chunks.append(current)
        return merged_chunks

if __name__ == "__main__":
    """
    Manual index builder / inspector.
    Run from terminal:
        python rag_pipeline.py
    """

    PDF_FOLDER = r"C:\Users\M.Asaf\Downloads\Agri PDFs\agri and waste water"
    INDEX_FILE = "pdf_index_enhanced.pkl"

    # Minimal model params – just enough to construct RAGPipeline.
    # We won't call .query() here, so the tokens/URL don't actually get used.
    model_params = {
        "llm_type": "deepseek",
        "hf_token": f'st.secrets["hf_backup_token_2"]',
        "deepseek_url": "https://router.huggingface.co/v1/chat/completions",
        "deepseek_model": "deepseek-ai/DeepSeek-V3.1:novita",
    }

    print("[MAIN] Initializing RAGPipeline...")
    pipeline = RAGPipeline(
        pdf_folder=PDF_FOLDER,
        index_file=INDEX_FILE,
        model_params=model_params,
    )

    # 1) Try to load existing index
    if pipeline.load_index():
        print("[MAIN] Existing index loaded.")
    else:
        print("[MAIN] No index found or failed to load. Building a new one...")
        try:
            total_chunks = pipeline.build_index()
            print(f"[MAIN] Index built successfully. Total chunks: {total_chunks}")
        except Exception as e:
            print("[MAIN] Index build failed:", e)
            raise

    # 2) Show a spread of sample chunks for inspection (max 1 per source PDF)
    # if pipeline.documents:
    #     print(f"\n[MAIN] Showing up to 20 sample chunks from different PDFs "
    #           f"(total chunks: {len(pipeline.documents)}):")

    #     seen_sources = set()
    #     shown = 0
    #     for doc in pipeline.documents:
    #         src = doc.metadata.get("source", "Unknown")
    #         if src in seen_sources:
    #             continue
    #         seen_sources.add(src)
    #         shown += 1

    #         print(f"\n--- Sample {shown} ---")
    #         print("source :", src)
    #         print("metadata:", doc.metadata)
    #         preview = doc.page_content[:400].replace("\n", " ")
    #         print("text    :", preview, "...")

    #         if shown >= 20:
    #             break
    # else:
    #     print("[MAIN] No documents in pipeline.documents after load/build.")

    # print("\n[MAIN] Debug: chunks for Rao-2018-Power_from_agro-waste-Business_Model_6.pdf")
    # pipeline.debug_print_chunks_for_source("Rao-2018-Power_from_agro-waste-Business_Model_6.pdf",
    #                                        max_chunks=20)

"""
[MAIN] Showing up to 20 sample chunks from different PDFs (total chunks: 4781):

--- Sample 1 ---
source : Danso-2018-Fixed_wastewater-freshwater_swap_Mashhad_Plain_Iran-Case_Study.pdf
metadata: {'source': 'Danso-2018-Fixed_wastewater-freshwater_swap_Mashhad_Plain_Iran-Case_Study.pdf', 'page': 1, 'type': 'heading'}
text    : CASE Fixed wastewater-freshwater swap (Mashhad Plain, Iran) Value offer: Treated wastewater for farmers in exchange for freshwater for urban use Organization type: Public and private (farmer associations) Status of organization: Operational (since 20052008) Major partners: Khorasan Razavi Regional W ...

--- Sample 2 ---
source : Drechsel-2018-Corporate_social_responsibility_CSR_as_driver_of_change-Business_Model_22.pdf
metadata: {'source': 'Drechsel-2018-Corporate_social_responsibility_CSR_as_driver_of_change-Business_Model_22.pdf', 'page': 1, 'type': 'heading', 'chunk_index': 0, 'total_chunks': 2}
text    : BUSINESS MODEL 22 Corporate Social Responsibility (CSR) as driver of change Model name Corporate Social Responsibility as driver of change Location of supporting cases Ghana (with additional input from other countries) Waste stream Wastewater ...

--- Sample 3 ---
source : Drechsel-2018-Wastewater_for_agriculture_forestry_and_aquaculture-Section_IV.pdf
metadata: {'source': 'Drechsel-2018-Wastewater_for_agriculture_forestry_and_aquaculture-Section_IV.pdf', 'page': 1, 'type': 'heading'}
text    : SECTION IV  WASTEWATER FOR AGRICULTURE, FORESTRY AND AQUACULTURE ...

--- Sample 4 ---
source : Drechsel-2018-Wastewater_for_fruit_and_wood_production_Egypt-Case_Study.pdf
metadata: {'source': 'Drechsel-2018-Wastewater_for_fruit_and_wood_production_Egypt-Case_Study.pdf', 'page': 1, 'type': 'heading'}
text    : CASE Wastewater for fruit and wood production (Egypt) Location: El-Gabal El-Asfar, northeast Cairo, Egypt Waste input type: Domestic and small industrial wastewater Value offer: Secondary treated wastewater reuse for cactus fruits (70%), lemon trees, and wood production; sludge sale for composting a ...

--- Sample 5 ---
source : Felgenhauer-2018-Enabling_private_sector_investment_in_large_scale_wastewater_treatment-Business_Model_19.pdf
metadata: {'source': 'Felgenhauer-2018-Enabling_private_sector_investment_in_large_scale_wastewater_treatment-Business_Model_19.pdf', 'page': 1, 'type': 'heading', 'chunk_index': 0, 'total_chunks': 2}
text    : BUSINESS MODEL 19 Enabling private sector investment in large scale wastewater treatment Model name Enabling private sector investment in large-scale wastewater treatment Value-Added Waste Products Treated wastewater for irrigation and a healthy environment Objective of entity Cost-recovery [ ]; For ...

--- Sample 6 ---
source : Gebreszgabher-2018-Bio-ethanol_and_chemical_products_from_agro_and_agro-industrial_waste-Business_Model_9.pdf
metadata: {'source': 'Gebreszgabher-2018-Bio-ethanol_and_chemical_products_from_agro_and_agro-industrial_waste-Business_Model_9.pdf', 'page': 1, 'type': 'heading', 'chunk_index': 0, 'total_chunks': 4}
text    : BUSINESS MODEL 9 Bio-ethanol and chemical products from agro and agro-industrial waste Model name Bio-ethanol and chemical products from agro and agro-industrial waste Waste stream Organic waste  Agro-waste from farms and agro-processing factories and vinasse waste generated during ethanol productio ...

--- Sample 7 ---
source : Gebrezgabher-2018-Combined_heat_and_power_and_ethanol_from_sugar_industry_waste_SSSSK_Maharashtra_India-Case_Study.pdf
metadata: {'source': 'Gebrezgabher-2018-Combined_heat_and_power_and_ethanol_from_sugar_industry_waste_SSSSK_Maharashtra_India-Case_Study.pdf', 'page': 1, 'type': 'heading'}
text    : CASE Combined heat and power and ethanol from sugar industry waste (SSSSK, Maharashtra, India) Waste input type: Bagasse, molasses, spent wash (Distillery efﬂuent) Value offer: Electricity/heat, ethanol, pressed mud and bio-fertilizer Status of organization: Operational (since 1962), co-generation u ...

--- Sample 8 ---
source : Gebrezgabher-2018-Combined_heat_and_power_from_agro-industrial_waste_for_on-and_off-site_use-Business_Model_8.pdf
metadata: {'source': 'Gebrezgabher-2018-Combined_heat_and_power_from_agro-industrial_waste_for_on-and_off-site_use-Business_Model_8.pdf', 'page': 1, 'type': 'heading', 'chunk_index': 0, 'total_chunks': 2}
text    : BUSINESS MODEL 8 Combined heat and power from agro-industrial waste for on- and off-site use Model name Combined heat and power from agro-industrial waste for on- and off-site use Waste stream Agro-industrial waste  Bagasse from sugar processing factories; Efﬂuent (solid and liquid waste) from agro- ...

--- Sample 9 ---
source : Hanjra-2018-Wastewater_as_a_commodity_driving_change-Business_Model_23.pdf
metadata: {'source': 'Hanjra-2018-Wastewater_as_a_commodity_driving_change-Business_Model_23.pdf', 'page': 1, 'type': 'heading', 'chunk_index': 0, 'total_chunks': 3}
text    : BUSINESS MODEL 23 Wastewater as a commodity driving change Munir A. Hanjra, Krishna C. Rao, George K ...

--- Sample 10 ---
source : IWMI-2022-Evidence-based_strategies_to_accelerate_innovation_scaling_in_agricultural_value_chains.pdf
metadata: {'source': 'IWMI-2022-Evidence-based_strategies_to_accelerate_innovation_scaling_in_agricultural_value_chains.pdf', 'page': 1, 'type': 'heading'}
text    : Evidence-based strategies to accelerate innovation scaling in agricultural value chains ...

--- Sample 11 ---
source : IWMI-2023-Developing_bankable_water_reuse_projects_guidelines_for_planners_investors_project_designers_and_operators.pdf
metadata: {'source': 'IWMI-2023-Developing_bankable_water_reuse_projects_guidelines_for_planners_investors_project_designers_and_operators.pdf', 'page': 1, 'type': 'heading', 'chunk_index': 0, 'total_chunks': 2}
text    : Guidelines for planners, investors, project designers and operators business value propositions, setting priorities, and developing a business plan  Identify water reuse options that safeguard the environment and human health, and those that provide products or services further downstream in the val ...

--- Sample 12 ---
source : Lebel-2018-Combined_heat_and_power_from_agro-industries_wastewater_TBEC_Bangkok_Thailand-Case_Study.pdf
metadata: {'source': 'Lebel-2018-Combined_heat_and_power_from_agro-industries_wastewater_TBEC_Bangkok_Thailand-Case_Study.pdf', 'page': 1, 'type': 'heading', 'chunk_index': 0, 'total_chunks': 2}     
text    : CASE Combined heat and power from agro-industrial wastewater (TBEC, Bangkok, Thailand) Waste input type: Wastewater from agricultural industries (starch, palm oil and ethanol) Value offer: Build, Own, Operate and Transfer (BOOT), one-stop shop to treat agro- industrial efﬂuent and generate electrici ...

--- Sample 13 ---
source : Mateo-Sagasta-2022-Water_reuse_in_the_Middle_East_and_North_Africa_a_sourcebook.pdf
metadata: {'source': 'Mateo-Sagasta-2022-Water_reuse_in_the_Middle_East_and_North_Africa_a_sourcebook.pdf', 'page': 1, 'type': 'heading'}
text    : Water reuse in the Middle East and North Africa A sourcebook Edited by: Javier Mateo-Sagasta (IWMI) Mohamed Al-Hamdi (FAO) Khaled AbuZeid (AWC) ...

--- Sample 14 ---
source : Mateo-Sagasta-2023-Expanding_water_reuse_in_the_Middle_East_and_North_Africa_policy_report.pdf
metadata: {'source': 'Mateo-Sagasta-2023-Expanding_water_reuse_in_the_Middle_East_and_North_Africa_policy_report.pdf', 'page': 1, 'type': 'heading'}
text    : Expanding water reuse in the Middle East and North Africa Policy Report Javier Mateo-Sagasta, Marie Helene Nassif, Mohamed Tawﬁk, Solomie Gebrezgabher, Everisto Mapedza, Nisreen Lahham and Mohamed Al-Hamdi ...

--- Sample 15 ---
source : Otoo-2018-Case_Agricultural_waste_to_high_quality_compost_DutuTech_Kenya.pdf
metadata: {'source': 'Otoo-2018-Case_Agricultural_waste_to_high_quality_compost_DutuTech_Kenya.pdf', 'page': 1, 'type': 'heading'}
text    : CASE Agricultural waste to high quality compost (DuduTech, Kenya) Miriam Otoo, Nancy Karanja, Jack Odero and Lesley Hope Waste input type: Vegetative waste, livestock waste Scale of businesses: Processes 125 tons of waste per month Major partners: Finlays Kenya Limited, Local livestock farmers ...

--- Sample 16 ---
source : Otoo-2018-Enriched_compost_production_from_sugar_industry_waste_PASIC_India-Case_Study.pdf
metadata: {'source': 'Otoo-2018-Enriched_compost_production_from_sugar_industry_waste_PASIC_India-Case_Study.pdf', 'page': 1, 'type': 'heading'}
text    : CASE Enriched compost production from sugar industry waste (PASIC, India) Miriam Otoo, Marudhanayagam Nageswaran, Lesley Hope and Priyanie Amerasinghe Value offer: Provision of enriched pressmud compost for agricultural production Scale of businesses: Processes 6,0009,000 tons of waste/year Major pa ...

--- Sample 17 ---
source : Otoo-2018-Nutrient_recovery_from_own_agro-industrial_waste-Business_Model_13.pdf
metadata: {'source': 'Otoo-2018-Nutrient_recovery_from_own_agro-industrial_waste-Business_Model_13.pdf', 'page': 1, 'type': 'heading', 'chunk_index': 0, 'total_chunks': 2}
text    : BUSINESS MODEL 13 Nutrient recovery from own agro-industrial waste Model name Nutrient recovery from own agro-industrial waste Value-Added Waste Products Regular compost, enriched vermi-compost Geography Regions with signiﬁcant livestock production, agro-processing enterprises Scale of production Me ...

--- Sample 18 ---
source : Rao-2018-Power_from_agro-waste-Business_Model_6.pdf
metadata: {'source': 'Rao-2018-Power_from_agro-waste-Business_Model_6.pdf', 'page': 1, 'type': 'paragraph', 'chunk_index': 0, 'total_chunks': 2}
text    : Waste stream Agro-waste (from farmers and agro-industries) Value-added product Power (through biomass gasiﬁcation or combustion) Geography Rural areas with large acres of crop cultivation for ease of procurement of crop residues Scale of production Small to medium scale; 25100 kW (gasiﬁcation) and u ...

--- Sample 19 ---
source : Watson-2018-Bio-ethanol_from_cassava_waste_ETAVEN_Carabobo_Venezuela-Case_Study.pdf
metadata: {'source': 'Watson-2018-Bio-ethanol_from_cassava_waste_ETAVEN_Carabobo_Venezuela-Case_Study.pdf', 'page': 1, 'type': 'heading'}
text    : CASE Bio-ethanol from cassava waste (ETAVEN, Carabobo, Venezuela) Value offer: Bio-ethanol (as additive to petrol/ gasoline as transportation fuel) Status of organization: Established in 2007, Business operational since 2012 Major partners: University of Carabobo, Ministry of Science and Technology, ...

--- Sample 20 ---
source : Watson-2018-Organic_binder_from_alcohol_production-Case_study.pdf
metadata: {'source': 'Watson-2018-Organic_binder_from_alcohol_production-Case_study.pdf', 'page': 1, 'type': 'heading'}
text    : CASE Organic binder from alcohol production (Eco Biosis S.A., Veracruz, Mexico) Waste input type: Vinasse waste (from alcohol production) Value offer: Clean water and chemical additive (for cement) Status of organization: Established in 2011, business operational since March 2013 Scale of businesses ...
"""


#It's actually correctly identifying headings and paragraphs.