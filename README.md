# FinRAG

Financial Domain-Specific RAG System using AWS Bedrock

A Retrieval-Augmented Generation (RAG) system for financial document analysis.

**Models**: AWS Bedrock Titan (embeddings + generation) + Google Gemini Pro (evaluation)

## 📄 Project Structure

```
FinRAG/
├── src/
│   ├── config.py          # Configuration management
│   ├── data_loader.py     # Load datasets
│   ├── embeddings.py      # Bedrock embeddings
│   ├── vector_store.py    # FAISS index
│   ├── retriever.py       # Document retrieval
│   ├── rag_pipeline.py    # RAG orchestration
│   └── evaluator.py       # Ragas evaluation
├── demo.py                # Interactive demo
├── main.py                # Batch evaluation
├── interactive_qa.py      # Interactive Q&A
├── config.yaml            # Configuration
└── requirements.txt       # Dependencies
```

## 🚀 Quick Start

### 1. Download Dataset

```bash
# Clone the FinanceRAG Challenge dataset
git clone https://github.com/charlieoneill11/icaif-24-finance-rag-challenge.git
```

**Note**: Dataset is ~68MB and not included in this repository.

### 2. Install Dependencies

```bash
cd FinRAG
pip install -r requirements.txt
```

### 3. Configure AWS Credentials

Create `.env` file:
```bash
cat > .env << 'EOF'
# AWS Credentials (for RAG system)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_SESSION_TOKEN=your_token  # if using temporary credentials
AWS_REGION=us-west-2

# Google Gemini API Key (for Ragas evaluation - get from https://aistudio.google.com/apikey)
GOOGLE_API_KEY=your_gemini_key
EOF
```

**Enable Bedrock models** in [AWS Console](https://console.aws.amazon.com/bedrock/):
- ✅ Amazon Titan Embeddings V2
- ✅ Amazon Titan Text Express (for answer generation)

### 4. Run Scripts

```bash
# Interactive demo (recommended first)
python demo.py

# Batch evaluation (retrieval only, fast)
python main.py --skip-generation --dataset financebench

# Full RAG pipeline (with answer generation)
python main.py --dataset financebench --max-queries 10

# Ask custom questions (searches all datasets)
python interactive_qa.py "What is Boeing's revenue in 2022?"
```

---

## 📊 Three Scripts Explained

| Script | Purpose | When to Use | Data Scope |
|--------|---------|-------------|------------|
| **demo.py** | Step-by-step demo + 10 test queries | First time, learning | **All 3 datasets** |
| **main.py** | Batch evaluation | Experiments, benchmarks | Configurable (single/all) |
| **interactive_qa.py** | Interactive Q&A | Ask your own questions | **All datasets by default** |

## 🔧 Detailed Command Usage

### 1️⃣ demo.py - Interactive Demo

**No arguments needed**, just run:

```bash
python demo.py
```

**What it does**:
- Loads **all 3 datasets** (finqa, financebench, finder) - 11,430 documents total
- Shows each step: load → embed → index → retrieve → generate
- Tests **10 custom queries** defined in Phase 1 Request Set:
  1. How much revenue does Microsoft generate from contracts with customers?
  2. When did Coupang's Farfetch consolidation start?
  3. What was the change in total expense net of tax for share based compensation from 2014 to 2015?
  4. Did abiomed outperform the nasdaq medical equipment index?
  5. ... (and 6 more questions)
- Displays answers and performance statistics for all 10 queries
- Shows average latency and success rate

**Output**: Terminal output with detailed results for each query and performance summary

**Note**: This demonstrates the system's ability to retrieve from a comprehensive knowledge base (all datasets) to answer domain-specific financial questions.

---

### 2️⃣ main.py - Batch Evaluation

**Full command syntax**:

```bash
python main.py [options]
```

**Common Options**:

| Option | Description | Example |
|--------|-------------|---------|
| `--dataset <name>` | Process specific dataset | `--dataset financebench` |
| `--skip-generation` | Retrieval only (no LLM, faster) | `--skip-generation` |
| `--max-queries <n>` | Limit number of queries | `--max-queries 50` |
| `--skip-embeddings` | Use cached embeddings | `--skip-embeddings` |
| `--rebuild-index` | Force rebuild vector index | `--rebuild-index` |
| `--use-ragas` | Enable Ragas evaluation | `--use-ragas` |
| `--log-level <level>` | Set log level (DEBUG/INFO/WARNING/ERROR) | `--log-level DEBUG` |
| `--config <path>` | Custom config file | `--config my_config.yaml` |

**Usage Examples**:

```bash
# Quick test: retrieval only on 10 queries
python main.py --dataset financebench --max-queries 10 --skip-generation

# Full evaluation with answer generation (50 queries)
python main.py --dataset financebench --max-queries 50

# Evaluate all datasets (takes ~40 min)
python main.py

# Use cached embeddings for faster testing
python main.py --dataset finqa --skip-embeddings --skip-generation

# Full RAG evaluation with Ragas metrics (uses AWS Bedrock)
python main.py --dataset financebench --use-ragas --max-queries 20
```

**Available Datasets**:
- `financebench` (180 docs, 150 queries)
- `finqa` (6,251 docs, 1,147 queries)
- `finder` (4,999 docs, 216 queries)
- `convfinqa`, `finqabench`, `tatqa`, `multiheirtt`

**Output Files** (in `outputs/`):
- `retrieval_results_<timestamp>.json` - Retrieved documents
- `rag_responses_<timestamp>.json` - Generated answers
- `evaluation_report_<timestamp>.csv` - Metrics summary
- `finrag.log` - Detailed logs

---

### 3️⃣ interactive_qa.py - Custom Questions

**Default behavior**: Searches across **ALL datasets** (32K+ documents) for comprehensive answers.

**Two modes**: Interactive or Command-line

**Mode A: Interactive** (recommended)

```bash
python interactive_qa.py
```

Then type your questions:
```
> What is Boeing's total revenue in 2022?
> Does Apple have positive working capital?
> What is PepsiCo's operating margin?
```

**Mode B: Command-line**

```bash
python interactive_qa.py "<question>" [options]
```

**Options**:

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--dataset <name>` | Dataset to search | `all` | `--dataset finqa` |
| `--top-k <n>` | Number of docs to retrieve | `5` | `--top-k 10` |
| `--no-contexts` | Hide retrieved contexts | Show | `--no-contexts` |

**Usage Examples**:

```bash
# Basic question (searches all datasets)
python interactive_qa.py "What is Boeing's revenue in 2022?"

# Search specific dataset only
python interactive_qa.py "What is the revenue growth rate?" --dataset finqa

# More comprehensive search with higher top-k
python interactive_qa.py "Does Apple have debt?" --top-k 10

# Quick answer without showing contexts
python interactive_qa.py "What is Microsoft's cash flow?" --no-contexts

# Interactive mode for multiple questions
python interactive_qa.py
> What is Microsoft's operating margin?
> How much did Adobe spend on R&D?
> What is Tesla's free cash flow?
```

**Output**: 
- Retrieved document snippets (with scores and source datasets)
- Generated answer based on contexts
- Real-time response

---

## Phase 1

**Domain**: Financial document analysis (10-K filings, earnings reports)

**Challenges**:
- Complex financial jargon
- Multi-page documents with cross-references
- Need for factual accuracy (high-stakes decisions)

**Solution**: RAG system combining retrieval (factual grounding) + LLM (natural synthesis)

### Request Set (10 Test Queries)

As part of Phase 1, we defined 10 representative financial queries to evaluate the RAG system:

1. How much revenue does Microsoft generate from contracts with customers?
2. When did Coupang's Farfetch consolidation start?
3. What was the change in total expense net of tax for share based compensation from 2014 to 2015 in millions?
4. Did abiomed outperform the nasdaq medical equipment index?
5. How much revenue does Microsoft generate from contracts with customers? (duplicate test)
6. When did Coupang's Farfetch consolidation start? (duplicate test)
7. What is CPNG's free cash flow?
8. What was the difference in percentage cumulative total shareholder return on masco common stock versus the s&p 500 index for the five year period ended 2017?
9. What is the growth rate in the balance of standby letters of credit from 2006 to 2007?
10. What is the percentage change in revenue generated from non-us currencies from 2015 to 2016?

**These queries are used in `demo.py`** to demonstrate the system's capabilities across different types of financial questions.

---

## 📂 Data Sources

- **FinanceBench**: 180 docs, 150 questions (33 companies)
- **FinQA**: 6,251 docs, 1,147 questions (numerical reasoning)
- **FinDER**: 4,999 docs, 216 questions (information retrieval)

From [ACM-ICAIF '24 FinanceRAG Challenge](https://github.com/charlieoneill11/icaif-24-finance-rag-challenge)

## 🏗️ Architecture

```
Query → Embedding → Vector Search → Top-K Docs → LLM → Answer
                      (FAISS)                    (Bedrock)
```

**Components**:
- **Embeddings**: AWS Bedrock Titan Embeddings V2 (`amazon.titan-embed-text-v2:0`)
- **Vector Store**: FAISS (cosine similarity)
- **LLM**: Amazon Titan Text Express (`amazon.titan-text-express-v1`)
- **Evaluation**: Ragas (Google Gemini Pro `gemini-pro`) + IR metrics (Recall@K, MRR)

---

## 🤖 Models Used

| Component | Model | Version | Provider | Purpose |
|-----------|-------|---------|----------|---------|
| **Embedding** | Titan Embeddings V2 | `amazon.titan-embed-text-v2:0` | AWS Bedrock | Convert text to 1024-dim vectors |
| **Generation** | Titan Text Express | `amazon.titan-text-express-v1` | AWS Bedrock | Generate answers from context |
| **Evaluation** | Gemini Pro | `gemini-pro` | Google AI | Evaluate answer quality (Ragas) |

**Why these models?**
- **Titan Embeddings V2**: High-quality embeddings, cost-effective
- **Titan Text Express**: Fast, accurate, good cost/performance balance
- **Gemini Pro**: Independent evaluator (avoids self-evaluation bias), free for students

## ⚙️ Configuration

Edit `config.yaml`:

```yaml
aws:
  region: "us-west-2"
  bedrock:
    embedding_model: "amazon.titan-embed-text-v2:0"  # Titan Embeddings V2 (1024-dim)
    llm_model: "amazon.titan-text-express-v1"        # Titan Text Express (fast & accurate)
    temperature: 0.1  # Lower = more factual (0.0-1.0)
    max_tokens: 4096  # Max response length

retrieval:
  top_k: 10          # Number of documents to retrieve
  
rag:
  max_contexts: 3    # Contexts sent to LLM
```

**Note**: Ragas evaluation automatically uses Google Gemini Pro (`gemini-pro`). No additional configuration needed.

## 📖 Usage Examples

### Demo Script
```bash
python demo.py
# Shows step-by-step: load data → embed → build index → retrieve → generate
```

### Batch Evaluation
```bash
# Quick test (retrieval only)
python main.py --skip-generation --dataset financebench

# Full evaluation (with answer generation)
python main.py --dataset financebench --max-queries 50

# All datasets
python main.py
```

### Custom Questions
```bash
# Interactive mode (searches all datasets)
python interactive_qa.py

# Command line (searches all datasets by default)
python interactive_qa.py "What is Boeing's total revenue in 2022?"

# Search specific dataset only
python interactive_qa.py "revenue growth rate" --dataset finqa
```

## 📊 Evaluation Metrics

**Retrieval Metrics**:
- Recall@K: % of relevant docs found in top-K
- Precision@K: % of retrieved docs that are relevant
- MRR: Mean Reciprocal Rank

**RAG Metrics** (Ragas framework):
- Context Precision: Relevance of retrieved contexts
- Context Recall: Coverage of relevant information
- Faithfulness: Answer grounded in context
- Answer Relevancy: Answer addresses the question

**Note**: Ragas evaluation uses **Google Gemini** for objective assessment. This avoids model self-evaluation bias.

## 📚 References

- [ACM-ICAIF '24 FinanceRAG Challenge](https://github.com/charlieoneill11/icaif-24-finance-rag-challenge)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Ragas Evaluation Framework](https://docs.ragas.io/)
- See `phase1_writeup.md` for detailed project proposal
