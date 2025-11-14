import os
import fitz
import re
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util, CrossEncoder
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


class ConversationManager:
    """Manages conversation history with model-aware token limits"""
    
    MODEL_LIMITS = {
        "deepseek": 64000,
        "azure": 128000,
    }
    
    def __init__(self, llm_type: str, reserve_tokens: int = 8000):
        self.llm_type = llm_type.lower()
        self.max_context = self.MODEL_LIMITS.get(self.llm_type, 64000)
        self.reserve_tokens = reserve_tokens
        self.available_for_history = self.max_context - self.reserve_tokens
        
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            logger.warning("Failed to load tiktoken, using fallback estimation")
            self.tokenizer = None
        
        self.history = []
        
        logger.info(f"ConversationManager initialized: {self.llm_type}, "
                   f"max={self.max_context}, available_for_history={self.available_for_history}")
    
    def count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            return len(text) // 4
    
    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        total = 0
        for msg in messages:
            total += self.count_tokens(msg["content"])
            total += 4
        return total
    
    def add_exchange(self, user_message: str, assistant_message: str):
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": assistant_message})
        self._truncate_to_fit()
    
    def _truncate_to_fit(self):
        current_tokens = self.count_messages_tokens(self.history)
        
        while current_tokens > self.available_for_history and len(self.history) > 2:
            removed = self.history[:2]
            self.history = self.history[2:]
            
            removed_tokens = self.count_messages_tokens(removed)
            current_tokens -= removed_tokens
            
            logger.info(f"Truncated conversation: removed {removed_tokens} tokens, "
                       f"remaining={current_tokens}/{self.available_for_history}")
        
        if current_tokens > self.available_for_history * 0.8:
            pairs = len(self.history) // 2
            logger.warning(f"Conversation history at 80% capacity: {current_tokens} tokens, {pairs} pairs")
    
    def get_history(self) -> List[Dict[str, str]]:
        return self.history.copy()
    
    def get_history_tokens(self) -> int:
        return self.count_messages_tokens(self.history)
    
    def clear(self):
        self.history = []
        logger.info("Conversation history cleared")
    
    def get_stats(self) -> Dict:
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


class PDFExtractor:
    """Handles PDF extraction with layout preservation"""
    
    def __init__(self):
        self.header_footer_margin = 50
        self.min_text_length = 50
        self.heading_font_threshold = 13
    
    def extract_pdf(self, pdf_path: str) -> List[Dict]:
        try:
            content_blocks = self._extract_with_layout(pdf_path)
            
            if not content_blocks:
                logger.warning(f"Layout extraction failed, using fallback for {pdf_path}")
                content_blocks = self._fallback_extraction(pdf_path)
            
            content_blocks = self._merge_blocks(content_blocks)
            
            return content_blocks
        
        except Exception as e:
            logger.error(f"PDF extraction failed for {pdf_path}: {e}")
            return []
    
    def _extract_with_layout(self, pdf_path: str) -> List[Dict]:
        doc = fitz.open(pdf_path)
        all_content = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_height = page.rect.height
            
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                
                bbox = block["bbox"]
                
                if (bbox[1] < self.header_footer_margin or 
                    bbox[3] > page_height - self.header_footer_margin):
                    continue
                
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
                
                avg_font_size = np.mean(font_sizes) if font_sizes else 11
                content_type = "heading" if avg_font_size > self.heading_font_threshold else "paragraph"
                
                all_content.append({
                    "text": text,
                    "page": page_num + 1,
                    "type": content_type,
                    "bbox": bbox
                })
            
            tables = self._extract_tables(page, page_num + 1)
            all_content.extend(tables)
        
        doc.close()
        return all_content
    
    def _extract_tables(self, page, page_num: int) -> List[Dict]:
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
        if not content_blocks:
            return []
        
        merged = []
        current_block = None
        
        for block in content_blocks:
            if block["type"] == "table":
                if current_block:
                    merged.append(current_block)
                    current_block = None
                merged.append(block)
                continue
            
            if current_block is None:
                current_block = block.copy()
                continue
            
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
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\b\d{1,3}\s*$', '', text)
        text = re.sub(r'[^\w\s\.\,\:\;\-\(\)\[\]\"\'\%\$\#\@\!\?\/\&\+\=\*]', '', text)
        text = re.sub(r'\s+([.,;:!?)])', r'\1', text)
        text = re.sub(r'([(])\s+', r'\1', text)
        return text.strip()


class RAGPipeline:
    """Main RAG pipeline with advanced relevance filtering"""
    
    def __init__(
        self,
        pdf_folder: str,
        index_file: str,
        model_params: dict,
        reserve_tokens: int = 8000,
        # Production-optimized thresholds
        similarity_threshold: float = 0.60,  # Minimum cosine similarity
        rerank_top_k: int = 12,              # Documents to rerank (reduced for speed)
        final_top_k: int = 3,                # Final documents to use (focused)
        min_combined_score: float = 0.50,    # NEW: Minimum final score (STRICT)
    ):
        self.pdf_folder = pdf_folder
        self.index_file = index_file
        
        # Relevance filtering parameters
        self.similarity_threshold = similarity_threshold
        self.rerank_top_k = rerank_top_k
        self.final_top_k = final_top_k
        self.min_combined_score = min_combined_score  # NEW: Store as instance variable
        
        llm_type = model_params.get("llm_type", "").lower()
        if llm_type not in ("azure", "deepseek"):
            raise ValueError("model_params['llm_type'] must be 'azure' or 'deepseek'")
        
        self.llm_type = llm_type
        
        self.conversation_manager = ConversationManager(
            llm_type=self.llm_type,
            reserve_tokens=reserve_tokens
        )
        
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
        else:
            if "hf_token" not in model_params:
                raise ValueError("hf_token is required for DeepSeek")
            self.hf_token = model_params["hf_token"]
            self.deepseek_url = model_params.get("deepseek_url",
                                "https://api.deepseek.com/v1/chat/completions")
            self.deepseek_model = model_params.get("deepseek_model", "deepseek-ai/DeepSeek-V3.1:novita")
        
        # Embedding model
        self.embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        self.embedding_model.eval()
        
        # Reranker model (FIXED: simplified initialization)
        try:
            self.reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,
                device="cpu"  # Change to "cuda" if GPU available
            )
            logger.info("Reranker: MiniLM-L-6-v2 loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}")
            self.reranker = None
        
        self.pdf_extractor = PDFExtractor()
        
        # Chunking strategy
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            is_separator_regex=False,
        )
        
        self.documents = []
        self.embeddings = None
        self.faiss_retriever = None
        self.bm25_retriever = None
        self.hybrid_retriever = None
        
        # Log configuration
        logger.info(f"RAG Pipeline initialized: sim_threshold={similarity_threshold}, "
                   f"rerank_top_k={rerank_top_k}, final_top_k={final_top_k}, "
                   f"min_combined_score={min_combined_score}")
    
    def build_index(self, progress_callback=None, status_callback=None):
        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.lower().endswith(".pdf")]
        
        if not pdf_files:
            raise ValueError("No PDF files found in folder")
        
        all_documents = []
        
        for i, pdf_file in enumerate(pdf_files):
            if status_callback:
                status_callback(f"Processing: {pdf_file} ({i+1}/{len(pdf_files)})")
            
            pdf_path = os.path.join(self.pdf_folder, pdf_file)
            
            try:
                content_blocks = self.pdf_extractor.extract_pdf(pdf_path)
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
        
        texts = [doc.page_content for doc in all_documents]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        self._build_retrievers(all_documents, texts, embeddings)
        
        self.documents = all_documents
        self.embeddings = embeddings
        self._save_index()
        
        if status_callback:
            status_callback(f"✅ Indexed {len(all_documents)} chunks from {len(pdf_files)} PDFs")
        
        return len(all_documents)
    
    def _create_chunks(self, content_blocks: List[Dict], pdf_name: str) -> List[Document]:
        documents = []
        
        for block in content_blocks:
            text = self.pdf_extractor.clean_text(block["text"])
            
            if len(text) < 50:
                continue
            
            if block["type"] == "table" or len(text) < 500:
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": pdf_name,
                        "page": block["page"],
                        "type": block["type"]
                    }
                ))
            else:
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
        faiss_index = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            embedding=lambda x: self.embedding_model.encode(x, normalize_embeddings=True),
            metadatas=[doc.metadata for doc in documents]
        )
        self.faiss_retriever = faiss_index.as_retriever(search_kwargs={"k": self.rerank_top_k})
        
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = self.rerank_top_k
        
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[self.faiss_retriever, self.bm25_retriever],
            weights=[0.85, 0.15]
        )
    
    def _save_index(self):
        with open(self.index_file, "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "embeddings": self.embeddings,
                "model": "BAAI/bge-base-en-v1.5"
            }, f)
    
    def load_index(self):
        if not os.path.exists(self.index_file):
            return False
        
        try:
            with open(self.index_file, "rb") as f:
                data = pickle.load(f)
            
            self.documents = data["documents"]
            self.embeddings = data["embeddings"]
            
            texts = [doc.page_content for doc in self.documents]
            self._build_retrievers(self.documents, texts, self.embeddings)
            
            logger.info(f"Loaded index with {len(self.documents)} chunks")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def _filter_by_similarity(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """Filter documents by semantic similarity threshold"""
        if not documents:
            return []
        
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        doc_texts = [doc.page_content for doc in documents]
        doc_embeddings = self.embedding_model.encode(
            doc_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        similarities = util.cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()
        
        doc_scores = []
        for doc, score in zip(documents, similarities):
            if score >= self.similarity_threshold:
                doc_scores.append((doc, float(score)))
        
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Similarity filtering: {len(documents)} → {len(doc_scores)} docs "
                   f"(threshold: {self.similarity_threshold})")
        
        return doc_scores
    
    def _rerank_documents(self, query: str, doc_scores: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Rerank documents using cross-encoder (FIXED: always apply sigmoid)"""
        if not self.reranker or not doc_scores:
            logger.info("Reranker disabled or no docs → return original scores")
            return doc_scores

        docs = [doc for doc, _ in doc_scores]
        pairs = [[query, doc.page_content] for doc in docs]

        logger.info(f"Reranker INPUT: {len(pairs)} query-doc pairs")

        try:
            raw_scores = self.reranker.predict(pairs)
            logger.info(f"Reranker RAW output shape: {np.array(raw_scores).shape}")
            logger.info(f"Reranker RAW scores (first 3): {raw_scores[:3]}")

            # ALWAYS apply sigmoid to normalize to 0-1 range
            raw_array = np.array(raw_scores).astype(float)
            probs = 1 / (1 + np.exp(-raw_array))
            
            logger.info(f"Reranker PROBS min={probs.min():.4f} max={probs.max():.4f} mean={probs.mean():.4f}")

            # Combine scores: 60% reranker, 40% similarity
            combined = []
            for idx, (doc, sim_score) in enumerate(doc_scores):
                rerank_prob = float(probs[idx])
                combined_score = 0.6 * rerank_prob + 0.4 * sim_score
                combined.append((doc, combined_score))

            combined.sort(key=lambda x: x[1], reverse=True)

            logger.info(f"Reranker DONE → top combined score={combined[0][1]:.4f}")
            return combined

        except Exception as e:
            logger.error(f"Reranker crashed: {e}", exc_info=True)
            return doc_scores
    
    def _deduplicate_documents(self, doc_scores: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Remove duplicate or highly similar chunks"""
        if len(doc_scores) <= 1:
            return doc_scores
        
        unique_docs = []
        seen_content = set()
        
        for doc, score in doc_scores:
            content_hash = hash(doc.page_content[:200])
            
            if content_hash not in seen_content:
                unique_docs.append((doc, score))
                seen_content.add(content_hash)
        
        if len(unique_docs) < len(doc_scores):
            logger.info(f"Deduplication: {len(doc_scores)} → {len(unique_docs)} docs")
        
        return unique_docs
    
    def query(self, question: str, top_k: int = None) -> Tuple[str, List[Document], Dict]:
        """Query with advanced relevance filtering (PRODUCTION VERSION)"""
        if self.hybrid_retriever is None:
            raise ValueError("Index not loaded. Call load_index() or build_index() first.")
        
        if top_k is None:
            top_k = self.final_top_k
        
        try:
            # Step 1: Initial retrieval
            retrieved_docs = self.hybrid_retriever.invoke(question)
            logger.info(f"Initial retrieval: {len(retrieved_docs)} documents")
            
            if not retrieved_docs:
                answer = "I couldn't find relevant information in the documents."
                return answer, [], {"relevance_scores": [], "reason": "no_documents"}
            
            # Step 2: Cap candidates for reranking
            candidates = retrieved_docs[:self.rerank_top_k]
            logger.info(f"Capped candidates for reranking: {len(candidates)}")
            
            # Step 3: Filter by similarity threshold (0.60)
            doc_scores = self._filter_by_similarity(question, candidates)
            
            if not doc_scores:
                answer = ("I found some documents, but none meet the relevance threshold. "
                         "Could you rephrase or provide more specific details?")
                return answer, [], {"relevance_scores": [], "reason": "below_similarity_threshold"}
            
            # Step 4: Rerank for better relevance
            doc_scores = self._rerank_documents(question, doc_scores)
            
            # Step 5: Deduplicate
            doc_scores = self._deduplicate_documents(doc_scores)
            
            # Step 6: Take top-k most relevant
            final_doc_scores = doc_scores[:top_k]
            
            # Step 7: CRITICAL - Filter by minimum combined score (0.50)
            final_doc_scores = [(doc, score) for doc, score in final_doc_scores 
                               if score >= self.min_combined_score]
            
            if not final_doc_scores:
                answer = ("I found some documents, but none are sufficiently relevant (all below 0.50 confidence). "
                         "Could you rephrase your question or provide more specific details?")
                return answer, [], {"relevance_scores": [], "reason": "below_min_combined_score"}
            
            final_docs = [doc for doc, score in final_doc_scores]
            relevance_scores = [score for doc, score in final_doc_scores]
            
            logger.info(f"Final selection: {len(final_docs)} documents, "
                       f"scores: {[f'{s:.3f}' for s in relevance_scores]}")
            
            # Generate answer
            answer = self._generate_answer_with_history(question, final_docs, relevance_scores)
            
            # Store in history
            self.conversation_manager.add_exchange(question, answer)
            
            # Return with enhanced metadata
            metadata = {
                "relevance_scores": relevance_scores,
                "num_candidates": len(retrieved_docs),
                "num_filtered": len(doc_scores),
                "threshold": self.similarity_threshold,
                "min_combined_score": self.min_combined_score,
                "avg_score": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
                "max_score": max(relevance_scores) if relevance_scores else 0,
                "min_score": min(relevance_scores) if relevance_scores else 0,
                "query_length": len(question.split()),
            }
            
            # Log quality assessment
            avg_score = metadata["avg_score"]
            quality = "EXCELLENT" if avg_score > 0.7 else "GOOD" if avg_score > 0.6 else "FAIR" if avg_score > 0.5 else "POOR"
            logger.info(f"Answer quality: {quality} (avg_score={avg_score:.3f})")
            
            return answer, final_docs, metadata
            
        except Exception as e:
            logger.error(f"Error in query method: {e}", exc_info=True)
            error_answer = f"I encountered an error while processing your query: {str(e)}"
            return error_answer, [], {"error": str(e)}
    
    def _generate_answer_with_history(self, question: str, context_docs: List[Document], 
                                     relevance_scores: List[float]) -> str:
        """Generate answer with relevance-weighted context"""
        
        context_parts = []
        for i, (doc, score) in enumerate(zip(context_docs, relevance_scores), 1):
            src = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            doc_type = doc.metadata.get("type", "text")
            
            relevance = "HIGH" if score > 0.7 else "MEDIUM" if score > 0.6 else "MODERATE"
            
            type_label = f"[{doc_type.upper()}] " if doc_type != "paragraph" else ""
            context_parts.append(
                f"[Source {i}: {src}, Page {page}] [Relevance: {relevance}] {type_label}\n{doc.page_content}\n"
            )
        
        context = "\n".join(context_parts)
        
        system_prompt = """You are the Circular Bioeconomy Decision Support Assistant (CBE-DSA) - an AI-powered chatbot developed to disseminate applied research and evidence-based insights from the International Water Management Institute (IWMI) and related partners.

Your primary goal is to help users, including policymakers, industry professionals, entrepreneurs, investors, and development partners, make informed, evidence-based decisions in the circular bioeconomy and sustainable waste management.

Role and Behaviour:
- Serve as a research-driven knowledge advisor, interpreting academic and technical content into concise, practical, and actionable insights.
- Focus on bridging science and implementation by translating findings into real-world relevance for business models, innovation, and policy design.
- Remain accurate, context-aware, and user-oriented, tailoring responses to the user's role (e.g., policymaker vs. entrepreneur).
- Use information strictly derived from the uploaded IWMI documents and clearly indicate when requested information cannot be found or falls outside the scope of the available materials.
- Maintain conversation context and refer to previous exchanges when relevant.

Tone and Communication Style:
- Professional, clear, neutral, and factual.
- Use plain language; cite sources when needed.
- Emphasize practical impact and innovation.

Response Format:
Provide the response using a clear, structured format. Begin with a brief Overview summarizing the key findings in one or two sentences. Then present the Key Points using concise bullet points (•), avoiding long paragraphs. Add an Implications section using short bullet points that highlight practical or policy relevance. If any comparative or quantitative data are available, present them in a Markdown table. Always cite sources using the format [Source X] immediately after each claim. Break lines appropriately to avoid long, dense text. Integrate insights from multiple documents whenever possible to ensure a comprehensive answer.

### CONVERSATION FLOW & ENGAGEMENT (MANDATORY)
- **Every response must end with a proactive, context-aware follow-up question or suggestion.**
- This keeps the conversation alive and guides the user toward deeper insights.

Information Use:
- For general/greeting questions, ignore retrieved content and respond from general knowledge.
- For research-specific queries, rely strictly on IWMI sources.

Your mission is to provide support to science-based decision-making and accelerate the adoption of optimized circular bioeconomy business models, guided by IWMI's validated research and expertise."""
        
        user_prompt = f"""CONTEXT FROM DOCUMENTS:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Check if the context contains relevant information to answer the question
2. If YES: Provide answer with [Source X] citations
3. If NO or PARTIAL: Clearly state what information is missing
4. Do NOT invent information not present in the context

ANSWER:"""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        history = self.conversation_manager.get_history()
        if history:
            messages.extend(history)
        
        messages.append({"role": "user", "content": user_prompt})
        
        try:
            if self.llm_type == "azure":
                resp = self.llm_client.chat.completions.create(
                    model=self.azure_deployment,
                    messages=messages,
                    max_tokens=2000,
                    temperature=0.1,
                )
                return resp.choices[0].message.content.strip()

            else:  # deepseek
                headers = {
                    "Authorization": f"Bearer {self.hf_token}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.deepseek_model,
                    "messages": messages,
                    "max_tokens": 2000,
                    "temperature": 0.1
                }

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