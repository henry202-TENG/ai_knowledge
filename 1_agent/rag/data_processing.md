# Data Processing for RAG

RAG 系統的數據處理流程，從原始文檔到向量索引的完整 pipeline。

---

## 1. 文檔載入

### 多格式支援

```python
class DocumentLoader:
    """文檔載入器"""

    LOADERS = {
        ".txt": "TextLoader",
        ".pdf": "PDFLoader",
        ".docx": "DocxLoader",
        ".md": "MarkdownLoader",
        ".html": "HTMLLoader",
        ".csv": "CSVLoader",
        ".json": "JSONLoader"
    }

    @staticmethod
    def load(file_path):
        """載入文檔"""

        from langchain_community.document_loaders import (
            TextLoader, PyPDFLoader,
            Docx2txtLoader, UnstructuredMarkdownLoader,
            BSHTMLLoader, CSVLoader
        )

        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
        elif ext == ".html":
            loader = BSHTMLLoader(file_path)
        elif ext == ".csv":
            loader = CSVLoader(file_path)
        else:
            loader = TextLoader(file_path)

        return loader.load()


# 批量載入
class DirectoryLoader:
    """目錄批量載入"""

    def __init__(self, directory, extensions=None):
        self.directory = directory
        self.extensions = extensions or [".txt", ".pdf", ".md"]

    def load_all(self):
        """載入所有文檔"""

        documents = []

        for ext in self.extensions:
            files = glob.glob(f"{self.directory}/**/*{ext}", recursive=True)

            for file in files:
                try:
                    docs = DocumentLoader.load(file)
                    for doc in docs:
                        doc.metadata["source"] = file
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {file}: {e}")

        return documents
```

---

## 2. 文本分塊

### 分塊策略

```python
class TextChunker:
    """文本分塊器"""

    @staticmethod
    def chunk_by_size(documents, chunk_size=500, chunk_overlap=50):
        """按大小分塊"""

        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        return splitter.split_documents(documents)

    @staticmethod
    def chunk_by_sentence(documents):
        """按句子分塊"""

        from langchain.text_splitter import NLTKTextSplitter

        splitter = NLTKTextSplitter(chunk_size=500)

        return splitter.split_documents(documents)

    @staticmethod
    def semantic_chunking(documents, embedder):
        """語意分塊"""

        # 1. 獲取所有句子嵌入
        all_sentences = []
        for doc in documents:
            sentences = split_into_sentences(doc.page_content)
            all_sentences.extend([
                {"text": s, "metadata": doc.metadata}
                for s in sentences
            ])

        # 2. 計算相似度
        embeddings = embedder.embed([s["text"] for s in all_sentences])

        # 3. 根據相似度分組
        chunks = []
        current_chunk = []

        for i, emb in enumerate(embeddings):
            if not current_chunk:
                current_chunk.append(all_sentences[i])
                continue

            # 計算與上一個的相似度
            prev_emb = embeddings[len(chunks) * len(all_sentences) // len(chunks)]

            if cosine_similarity(emb, prev_emb) > 0.8:
                current_chunk.append(all_sentences[i])
            else:
                # 新塊
                chunks.append(merge_to_document(current_chunk))
                current_chunk = [all_sentences[i]]

        return chunks
```

### 結構化文檔

```python
class StructuredChunker:
    """結構化分塊"""

    @staticmethod
    def chunk_by_heading(documents):
        """按標題分塊"""

        from langchain.text_splitter import MarkdownHeaderTextSplitter

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3")
        ]

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )

        return splitter.split_text(documents[0].page_content)


# Markdown 分塊範例
MARKDOWN_SPLITTER = """
# 標題 1

內容 1

## 標題 2

內容 2

↓ 分塊結果:
[
  {"content": "標題 1\n\n內容 1", "metadata": {"level": 1}},
  {"content": "## 標題 2\n\n內容 2", "metadata": {"level": 2}}
]
"""
```

---

## 3. 向量化

### 嵌入模型

```python
class EmbeddingManager:
    """嵌入管理器"""

    MODELS = {
        "openai": "text-embedding-ada-002",
        "cohere": "embed-multilingual-v3.0",
        "local": "sentence-transformers/all-MiniLM-L6-v2",
        "bge": "BAAI/bge-small-zh-v1.5"
    }

    def __init__(self, model_name="openai"):
        self.model_name = model_name
        self.model = self._load_model(model_name)

    def _load_model(self, name):
        """加載模型"""

        if name == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=self.MODELS["openai"])

        elif name == "local":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=self.MODELS["local"]
            )

        elif name == "bge":
            from langchain_community.embeddings import HuggingFaceBgeEmbeddings
            return HuggingFaceBgeEmbeddings(
                model_name=self.MODELS["bge"]
            )

    def embed_documents(self, documents):
        """嵌入文檔"""

        texts = [doc.page_content for doc in documents]

        return self.model.embed_documents(texts)

    def embed_query(self, query):
        """嵌入查詢"""

        return self.model.embed_query(query)
```

---

## 4. 索引構建

### 向量存儲

```python
class VectorStoreBuilder:
    """向量存儲構建器"""

    @staticmethod
    def build_chroma(documents, embeddings, collection_name="rag"):
        """ChromaDB"""

        from langchain_community.vectorstores import Chroma

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name
        )

        return vectorstore

    @staticmethod
    def build_pinecone(documents, embeddings, index_name="rag"):
        """Pinecone"""

        from langchain_community.vectorstores import Pinecone

        vectorstore = Pinecone.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=index_name
        )

        return vectorstore

    @staticmethod
    def build_faiss(documents, embeddings):
        """FAISS"""

        from langchain_community.vectorstores import FAISS

        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embeddings
        )

        return vectorstore


# 完整 Pipeline
class RAGPipelineBuilder:
    """RAG Pipeline 構建"""

    def __init__(self):
        self.loader = None
        self.chunker = None
        self.embedder = None
        self.vectorstore = None

    def load_documents(self, path):
        """載入文檔"""
        self.loader = DirectoryLoader(path)
        return self

    def split_documents(self, chunk_size=500):
        """分塊"""
        docs = self.loader.load_all()
        self.chunker = TextChunker()
        self.documents = self.chunker.chunk_by_size(docs, chunk_size)
        return self

    def embed_and_index(self, model="openai"):
        """嵌入並索引"""
        self.embedder = EmbeddingManager(model)
        self.vectorstore = VectorStoreBuilder.build_chroma(
            self.documents,
            self.embedder.model
        )
        return self

    def build(self):
        """構建完成"""
        return {
            "vectorstore": self.vectorstore,
            "embedder": self.embedder,
            "documents": self.documents
        }
```

---

## 5. 數據增強

### 數據增強策略

```python
class DataAugmentation:
    """數據增強"""

    @staticmethod
    def paraphrase(texts, llm):
        """改寫增強"""

        augmented = []

        for text in texts:
            prompt = f"用不同的方式表達:\n{text}"

            new_text = llm.generate(prompt)

            augmented.append({
                "page_content": new_text,
                "metadata": {"augmented": True}
            })

        return augmented

    @staticmethod
    def back_translate(texts, source_lang="zh", target_lang="en"):
        """回譯增強"""

        # 翻譯成目標語言
        translated = translate_batch(texts, target_lang)

        # 翻譯回來
        back_translated = translate_batch(translated, source_lang)

        return back_translated
```

---

## 6. 品質控制

### 品質檢查

```python
class DataQualityChecker:
    """數據品質檢查"""

    def __init__(self):
        self.issues = []

    def check(self, documents):
        """檢查文檔品質"""

        for i, doc in enumerate(documents):
            # 1. 長度檢查
            if len(doc.page_content) < 50:
                self.issues.append({
                    "index": i,
                    "type": "too_short",
                    "length": len(doc.page_content)
                })

            # 2. 重複檢查
            if self._is_duplicate(doc, documents[:i]):
                self.issues.append({
                    "index": i,
                    "type": "duplicate"
                })

            # 3. 編碼檢查
            try:
                doc.page_content.encode('utf-8')
            except:
                self.issues.append({
                    "index": i,
                    "type": "encoding_error"
                })

        return {
            "total": len(documents),
            "issues": len(self.issues),
            "issue_list": self.issues
        }

    def _is_duplicate(self, doc, others):
        """檢查重複"""
        for other in others:
            if doc.page_content == other.page_content:
                return True
        return False
```

---

## 7. 與相關技術

| 技術 | 關係 |
|------|------|
| **VectorDB** | 向量存儲 |
| **RAG** | 檢索增強 |
| **LangChain** | 整合框架 |

---

## 延伸閱讀

- [LangChain Loaders](https://python.langchain.com/docs/modules/data_loader/)
- [Text Splitters](https://python.langchain.com/docs/modules/data_loader/document_splitners/)
- [ChromaDB](https://docs.trychroma.com/)