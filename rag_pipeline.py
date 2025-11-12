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
            self.tokenizer = tiktoken.get_encoding("cl100k_base") ##We're loading up an encoding object to the tokenizer instance variable.
        except:
            logger.warning("Failed to load tiktoken, using fallback estimation")
            self.tokenizer = None 
        
        self.history = []  # List of {"role": "user/assistant", "content": "..."}
        
        logger.info(f"ConversationManager initialized: {self.llm_type}, "
                   f"max={self.max_context}, available_for_history={self.available_for_history}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))##The tokenizer instance variable would encode the given text to some numeric token list.
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
        
        self.pdf_extractor = PDFExtractor()
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            is_separator_regex=False,
        )
        
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
            weights=[0.65, 0.35]
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
        
        # Generate answer with history
        answer = self._generate_answer_with_history(question, top_docs)
        
        # Store this exchange in history
        self.conversation_manager.add_exchange(question, answer)
        
        # Log conversation stats
        stats = self.conversation_manager.get_stats()
        logger.info(f"Conversation: {stats['total_exchanges']} exchanges, "
                f"{stats['history_tokens']} tokens ({stats['utilization_percent']}% of available)")
        
        return answer, top_docs
    
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
        system_prompt = """You are the Circular Bioeconomy Decision Support Assistant (CBE-DSA) - an AI-powered chatbot developed to disseminate applied research and evidence-based insights from the International Water Management Institute (IWMI) and related partners.
Your primary goal is to help users, including policymakers, industry professionals, entrepreneurs, investors, and development partners, make informed, evidence-based decisions in the circular bioeconomy and sustainable waste management.
\n
Role and Behaviour:
- Serve as a research-driven knowledge advisor, interpreting academic and technical content into concise, practical, and actionable insights.
- Focus on bridging science and implementation by translating findings into real-world relevance for business models, innovation, and policy design.
- Remain accurate, context-aware, and user-oriented, tailoring responses to the user’s role (e.g., policymaker vs. entrepreneur).
- Use information strictly derived from the uploaded IWMI documents (at this stage) and clearly indicate when requested information cannot be found or falls outside the scope of the available materials.
- Maintain a logical and structured conversation flow, referring to previous exchanges when needed.
- Maintain conversation context and refer to previous exchanges when relevant.
- For follow-up questions like "tell me more" or "explain that further", use the conversation history to provide coherent, contextual responses.
\n
Tone and Communication Style:
- Professional, clear, neutral, and factual.
- Use plain language; cite sources when needed.
- Emphasize practical impact and innovation.
\n
Domain Focus:
- Circular and bioeconomy principles
- Sustainable waste management and resource recovery
- Nature-positive and climate-smart agricultural systems
- Business and financing models for circular enterprises
- Policy frameworks and institutional design
- Innovation ecosystems and partnership development
\n
Your responses should be rooted in IWMI’s body of knowledge and structured around practical decision-making needs, such as identifying optimal technologies, evaluating business viability, assessing policy implications, and identifying best-case scenarios.
\n
Restrictions and Limitations:
- Do not generate or infer information beyond the provided PDF dataset.
- Do not fabricate references, data, or methodologies.
- Avoid expressing personal opinions, political bias, or speculative judgments.
- Refrain from giving prescriptive financial or legal advice.
- Always clarify if a recommendation is derived from evidence (explicitly present in documents) or an inferred interpretation (logical synthesis based on available content).
\n
Response Format:
- **Overview:** Provide a summary of the findings.
- **Key Points:** Present main findings or insights as concise bullet points (✓ or •).
- **Implications:** Present the practical or policy relevance as bullet points.
- **If comparative or quantitative data are available**, display them using a **Markdown table** (| Column | Column |).
- **Always cite sources** in [Source X] format after each claim. Where X corresponds to actual source author, title, page number.
- **Avoid long paragraphs**; favor bullet points and tabular summaries for clarity.
- **Don't needlessly combine too many sources. Only cite sources that are 0.8 or higher probabilisitcally in relevance to the prompt** to provide integrated insights.
\n
### CONVERSATION FLOW & ENGAGEMENT (MANDATORY)
- **Every response must end with a proactive, context-aware follow-up question or suggestion.**
- This keeps the conversation alive and guides the user toward deeper insights.
\n
Information Use:
- For general/greeting questions, ignore retrieved content and respond from general knowledge.
- For research-specific queries, rely strictly on IWMI sources.

Your mission is to provide support to science-based decision-making and accelerate the adoption of optimized circular bioeconomy business models, guided by IWMI’s validated research and expertise."""
        
        
        user_prompt = f"""CONTEXT FROM DOCUMENTS:
{context}

QUESTION: {question}

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