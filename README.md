# Agentic RAG — How `ingestion.py` works

## 1. One-liner

This script downloads 3 blog posts, chops them into small text chunks, turns each chunk into a number-vector (embedding), and stores those vectors in Pinecone so an agent can search them later.

---

## 2. Big-picture pipeline

```text
  urls (3 blog links)
           |
           v
  +------------------+
  | WebBaseLoader    |  download each page → Document(s)
  +------------------+
           |
           v
  +------------------+
  | flatten          |  [[docs],[docs],[docs]] → [doc, doc, ...]
  +------------------+
           |
           v
  +------------------+
  | text splitter    |  long pages → ~250-token chunks
  +------------------+
           |
           v
  +------------------+
  | OpenAIEmbeddings |  each chunk → 1024 numbers (a vector)
  +------------------+
           |
           v
  +------------------+
  | Pinecone upsert  |  store vectors in index "langgraph-rag"
  +------------------+
           |
           v
  +------------------+
  | as_retriever()   |  a search handle for later Q&A
  +------------------+
```

**Inputs → Outputs (whole file)**

| When | In | Out |
|------|----|-----|
| **You run the script** | 3 URLs + API keys in `.env` | Chunks embedded and stored in Pinecone; a `retriever` object in memory |
| **Later (other code)** | A user question string | Matching text chunks from the index |

---

## 3. Line-by-line walkthrough

### Step A — Load secrets (`load_dotenv`)

```python
load_dotenv()
```

- **What it is:** Reads your `.env` file and puts keys into environment variables.
- **In → out:** File on disk (e.g. `OPENAI_API_KEY`, `PINECONE_API_KEY`) → available to OpenAI / Pinecone clients. You don’t use a return value.
- **Why:** So you don’t hard-code secrets in the script.

---

### Step B — Pick the sources (`urls`)

```python
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
```

- **What it is:** A plain Python list of three web addresses (Lilian Weng blog posts).
- **In → out:** Nothing yet — just data waiting to be loaded.
- **Why:** These pages become the knowledge base the RAG agent can search.

---

### Step C — Download pages (`WebBaseLoader`)

```python
docs = [WebBaseLoader(url).load() for url in urls]
```

- **What it is:** For each URL, create a loader and call `.load()`.
- **In → out:**
  - **In:** one URL string
  - **Out:** a **list** of LangChain `Document` objects (usually one per page)
- **A `Document` is roughly:** `{ page_content: "the text...", metadata: {source: "url", ...} }`

```text
  url 1  -->  [Document(...)]
  url 2  -->  [Document(...)]
  url 3  -->  [Document(...)]

  docs = [ [Doc], [Doc], [Doc] ]   ← list of lists
```

- **Weird syntax:** `[... for url in urls]` = list comprehension — “do this for every url and collect the results.”

---

### Step D — Flatten to one list

```python
docs_list = [item for sublist in docs for item in sublist]
```

- **What it is:** Un-nests the list-of-lists into one flat list of documents.
- **In → out:** `[[A],[B],[C]]` → `[A, B, C]`

```text
  before:  docs      = [ [Doc1], [Doc2], [Doc3] ]
  after:   docs_list = [ Doc1, Doc2, Doc3 ]
```

- **Why:** The splitter wants a single list of documents, not a list of lists.

---

### Step E — Chop into chunks (`RecursiveCharacterTextSplitter`)

```python
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)
```

- **What it is:** A tool that breaks long text into smaller pieces measured in **tokens** (tiktoken ≈ how OpenAI counts text length).
- **In → out:**
  - **In:** full blog `Document`s
  - **Out:** many smaller `Document`s (~250 tokens each, no overlap)

```text
  LONG PAGE
  |------------------------------|
           split
  |----| |----| |----| |----| ...
   c1     c2     c3     c4
```

- **Why chunk?**
  - Embeddings work better on focused snippets.
  - Retrieval returns small relevant pieces, not whole essays.
- **Weird bits:**
  - `chunk_size=250` — target size per chunk (tokens)
  - `chunk_overlap=0` — chunks don’t share trailing/leading text
  - `from_tiktoken_encoder` — size by tokens, not raw characters

---

### Step F — Create the embedding model (`OpenAIEmbeddings`)

```python
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024,
)
```

- **What it is:** A client that turns text into a list of numbers (a **vector**) that capture meaning.
- **In → out (when used later):**
  - **In:** a string like `"agents use tools"`
  - **Out:** 1024 floats, e.g. `[0.01, -0.22, 0.55, ...]`
- **Why:** Similar meaning → similar vectors → Pinecone can find “close” chunks for a question.
- **Note:** This line only **builds** the embedding object. Embedding each chunk happens inside `from_documents` next.

```text
  "hello agents"  -->  [ 0.12, -0.44, 0.09, ... ]  (1024 numbers)
```

---

### Step G — Store in Pinecone (`PineconeVectorStore.from_documents`)

```python
vectorstore = PineconeVectorStore.from_documents(
    documents=doc_splits,
    embedding=embeddings,
    index_name="langgraph-rag",
)
```

- **What it is:** Embed every chunk and **upsert** (upload) those vectors into a Pinecone index.
- **In → out:**
  - **In:** chunk `Document`s + embedding model + index name
  - **Out:** a `vectorstore` object pointing at that index; data now lives in Pinecone cloud

```text
  chunk text  -->  embed  -->  vector + text  -->  Pinecone
                                              index: "langgraph-rag"
```

- **Why:** So later you can search by meaning (“what is prompt engineering?”) without re-downloading the blogs every time.

---

### Step H — Make a retriever

```python
retriever = vectorstore.as_retriever()
```

- **What it is:** A thin wrapper around the vector store with a simple search API.
- **In → out (when you invoke it later):**
  - **In:** a query string
  - **Out:** a list of relevant `Document` chunks
- **Example later:** `retriever.invoke("What is an agent?")` → top matching chunks

```text
  "What is an agent?"  -->  retriever  -->  [chunk, chunk, ...]
```

---

## 4. Filled mini-example (fake, tiny)

Imagine one tiny page instead of three blogs:

1. **Load:** URL → `Document(page_content="Agents can use tools to search the web.")`
2. **Split:** already short → maybe 1 chunk (same text)
3. **Embed:** that sentence → `[0.02, -0.11, ..., ]` length 1024
4. **Store:** vector + text land in Pinecone index `langgraph-rag`
5. **Retrieve later:** question `"Do agents use tools?"` → that chunk comes back because vectors are close

---

## 5. Not happening yet

This file **only fills** the vector database and builds a `retriever` in memory.

It does **not**:

- Answer user questions
- Run a LangGraph agent
- Call or replace `main.py` (that’s still a stub)

Think of ingestion as **stocking the library**. The agent (later) is the **librarian** who looks things up.

---

## 6. How to run

1. Put keys in `.env` (at least OpenAI + Pinecone).
2. Create a Pinecone index named **`langgraph-rag`** with dimension **`1024`** (must match `OpenAIEmbeddings`).
3. Run once to populate:

```bash
uv run python ingestion.py
```

After that, other code can search the same index without re-ingesting every time (unless you change the source URLs or want a fresh upload).
