# KI-Kompass: RAG Prototype

A collaborative prototyping project building an open-source RAG (Retrieval-Augmented Generation) backbone that SMEs can adapt for their own knowledge management.

## Scenario
**PrimePack AG** sells packaging products (pallets, cardboard boxes, tape) sourced from multiple suppliers. Sustainability claims are increasingly important, from customers requiring CSRD-compliant data to procurement needing to evaluate supplier evidence quality.

The core challenge: sustainability information is spread across many documents (EPDs, supplier brochures, company reports, regulatory frameworks), claims have widely varying evidence quality, and employees must answer questions like:

- *"What does Supplier A claim for Product X, and can we trust it?"*
- *"Which suppliers are compliant with our EPD requirement?"*
- *"Can we tell a customer that this tape is PFAS-free?"*
- *"Which evidence is missing before we can respond?"*

The RAG assistant should **cite sources explicitly**, **separate facts from claims**, **highlight gaps and conflicts**, and **refuse to conclude when evidence is insufficient**.

---

## Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) (it installs Python 3.12+ and all dependencies for you).

```bash
# Create .venv and install everything (toolkit, backend, notebook tools) from uv.lock
uv sync
```

The `conversational_toolkit` and `ki_kompass_rag` packages are installed into `.venv` as editable workspace members. Run commands with `uv run ...`, or activate the environment with `source .venv/bin/activate` (on Windows: `.venv\Scripts\activate`).

To add or update a dependency, edit the relevant `pyproject.toml` and run `uv lock` followed by `uv sync`.

### Ollama (local LLM)

[Ollama](https://ollama.com) lets you run open-weight language models locally -> no API key, no data sent externally. Use it as a privacy-preserving alternative to the OpenAI backend.

**Install Ollama**

| Platform | Command                                                                                  |
|----------|------------------------------------------------------------------------------------------|
| macOS    | `brew install ollama` or download the app from [ollama.com](https://ollama.com/download) |
| Linux    | `curl -fsSL https://ollama.com/install.sh \| sh`                                         |
| Windows  | Download the installer from [ollama.com](https://ollama.com/download)                    |

**Start the server and download a model**

```bash
# Start the Ollama server (runs in the background on port 11434)
ollama serve

# Pull the default workshop model (~7 GB)
ollama pull mistral-nemo:12b

# List models already downloaded
ollama list
```

The server must be running whenever you use `BACKEND=ollama`. On macOS the desktop app starts the server automatically; on Linux run `ollama serve` in a separate terminal.

---

## Running the Notebooks (JupyterLab)

The workshop notebooks require JupyterLab. You can run them in an IDE or start it from the project root with uv:

```bash
uv run jupyter lab
```

JupyterLab opens in your browser at `http://localhost:8888`. Navigate to `backend/notebooks/english/` in the file browser on the left and open `T01_baseline_rag.ipynb` to start.

**Running cells:** Use `Shift+Enter` to run a cell and advance to the next one. Run cells top to bottom, each notebook must be executed in order within a session.

**OpenAI API key:** If you want to use OpenAI models, you need to set an OpenAI API key. Set the key before starting JupyterLab, or export it inside a cell:

```bash
export OPENAI_API_KEY="sk-..."
uv run jupyter lab
```

### Running the Notebooks on Renku

In a workshop you get a link to a Renku project. Start a session there: everything is already installed and the OpenAI key is set up for you, so you can open `backend/notebooks/english/T01_baseline_rag.ipynb` right away. Organisers set this up following [docs/renku_workshop_setup.md](docs/renku_workshop_setup.md).

---

## Quick Start

### Running the full stack (backend API + frontend)

This launches the FastAPI backend and the Next.js chat frontend together.

**Prerequisites:** Python ≥ 3.12, Node.js / npm, and either an OpenAI API key or a running Ollama server (see [Ollama](#ollama-local-llm) below).

**Step 1: Install Python dependencies** (skip if already done)

```bash
uv sync
```

**Step 2: Create vector stores**

Build the text and image vector stores the chat app uses (run once, after creating `.env.local` in Step 3). The image store downloads the Qwen3-VL embedding model and is slow on CPU; add `--skip-images` to build only the text store.

```bash
uv run python -m ki_kompass_rag.build_app_db
```

**Step 3: Create `.env.local`** (required for `make run`)

The Makefile reads your API key from a `.env.local` file in the project root. Create it with:

```bash
echo "OPENAI_API_KEY=sk-..." > .env.local
```

Or open the file in an editor and add:

```
OPENAI_API_KEY=sk-...
```

> If you run the backend without the Makefile, you can export the key in your shell instead and skip this file — see the "Without the Makefile" command in Step 3.

**Step 4: Start the backend**

With the Makefile:

```bash
make run
```

Without the Makefile:

```bash
# OpenAI
BACKEND=openai uv run python -m ki_kompass_rag.main

# Ollama (requires `ollama serve` and a pulled model)
BACKEND=ollama uv run python -m ki_kompass_rag.main
```

The server starts at `http://localhost:8080`.

**Step 5: Start the frontend** (in a separate terminal)

```bash
cd frontend
npm install # first time only
npm run dev
```

**Step 6: Open the app**

Go to `http://localhost:3000` in your browser.

---

### Notebooks (exploration and learning)

```bash
jupyter lab
# Open backend/notebooks/english/T01_baseline_rag.ipynb to start
```

### Single query from the command line

```bash
BACKEND=openai uv run python -m ki_kompass_rag.baseline_rag
BACKEND=openai QUERY="Which tape products have a verified EPD?" python -m ki_kompass_rag.baseline_rag
BACKEND=openai RESET_VS=1 python -m ki_kompass_rag.baseline_rag  # rebuild vector store
```

---

## Workshop Feature Tracks

The workshop is structured as a sequence of numbered notebooks (`T01` to `T14`) in `backend/notebooks/english/`. Run notebooks top to bottom within a session.

Start with `T01_baseline_rag.ipynb`. It builds the notebooks' vector store (`backend/db/data_vs.db`) on its first run; no separate setup step is needed. The notebooks import the `conversational_toolkit` library.

| Notebook                                                 | Topic                            | What you learn                                                                                                                                                                    |
|----------------------------------------------------------|----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `T01_baseline_rag.ipynb`                           | **Baseline RAG Pipeline**        | The five-stage RAG loop (chunk, embed, store, retrieve, generate), how retrieval inspection works as a debugging tool, and the main failure modes the workshop addresses.         |
| `T02_ingestion_chunking.ipynb`                              | **Document Ingestion**           | PDF parser comparison (markitdown vs. docling), chunking strategies (fixed-size, header-based, paragraph-aware), token limits, and how chunking choice affects retrieval quality. |
| `T03_multimodal_rag.ipynb`                                  | **Multimodal RAG**              | Retrieving text and images from a PDF with one pipeline (Qwen3-VL image embeddings). |
| `T04_evaluation.ipynb`                             | **Evaluation & Validation**      | Reference-free RAGAS metrics (Faithfulness, AnswerRelevancy); ground-truth metrics (ContextPrecision, ContextRecall); systematic comparison of RAG pipeline configurations.       |
| `T05_synthetic_dataset.ipynb`                       | **Synthetic Dataset Generation** | Building a validation dataset from vector store content; LLM-generated Q&A pairs; scaling beyond manual ground-truth creation.                                                    |
| `T06_structured_outputs.ipynb`                     | **Structured Outputs**           | Entity-grounding prompts to prevent hallucination; structured JSON outputs with evidence levels (VERIFIED / CLAIMED / MISSING / MIXED); making claim quality machine-readable.    |
| `T07_advanced_retrieval.ipynb`                     | **Retrieval Strategies**         | BM25 keyword retrieval; hybrid search via Reciprocal Rank Fusion (RRF); metadata filtering to scope retrieval to known documents.                                                 |
| `T08_conversation.ipynb`                           | **Conversational RAG**           | Maintaining conversation history; multi-turn context management.                                                                                                                  |
| `T09_query_techniques.ipynb`                       | **Query Improvement**            | Query expansion and HyDE (Hypothetical Document Embeddings) to improve semantic retrieval.                                                                                        |
| `T10_tools.ipynb` – `T14_rag_subagent.ipynb` | **Tools & Agents**               | LLM tool use; RAG as a tool; multi-step agent workflows; supplier comparison and evidence audit patterns.                                                                         |

---

## Repository Structure

```
.
├── backend/                                # RAG pipeline application
│   ├── notebooks/                          # Workshop notebooks
│   │   ├── english/                        # T01_baseline_rag ... T14_rag_subagent
│   │   └── german/                         # German versions (same file names)
│   └── src/ki_kompass_rag/
│       ├── db/
│       │   ├── vs_text/                    # ChromaDB collection: text chunks (OpenAI embeddings)
│       │   └── vs_image/                   # ChromaDB collection: image chunks (Qwen3-VL embeddings)
│       ├── baseline_rag.py        # Five-step pipeline (chunking, embedding, retrieval, generation)
│       ├── build_app_db.py        # Builds the chat app's text and image vector stores
│       ├── ingestion.py           # Parser comparison, chunking utilities, token analysis
│       ├── evaluation.py          # Shared evaluation queries and ground-truth answers
│       ├── advanced_retrieval.py
│       └── main.py                         # FastAPI server entry point
│
├── conversational-toolkit/                 # Reusable RAG components (toolkit library)
│   └── src/conversational_toolkit/
│       ├── agents/                         # RAG agent (retrieval + generation)
│       ├── api/                            # FastAPI server and routes
│       ├── chunking/                       # PDF, Excel, Markdown chunkers
│       ├── embeddings/                     # Sentence Transformer embeddings
│       ├── llms/                           # OpenAI, Ollama, local LLM backends
│       ├── retriever/                      # Vector store retriever, BM25, hybrid
│       ├── utils/                          # Prompt building, query expansion, RRF
│       └── vectorstores/                   # ChromaDB vector store
│
├── data/                                   # Document corpus
├── docs/                                   # Organiser guides (Renku workshop setup)
├── init.d/                                 # Renku session start script (sets OPENAI_API_KEY)
│
└── frontend/                               # Next.js chat frontend
```

---

## Dataset

All documents are in `data/`. The file prefix defines the document type:

| Prefix        | Meaning                                                        |
|---------------|----------------------------------------------------------------|
| `ART_`        | Artificial workshop scenario document                          |
| `EPD_`        | Verified, third-party Environmental Product Declaration        |
| `SPEC_`       | Product specification, datasheet, or article document          |
| `REF_`        | Regulatory or reference document                               |
| `EVALUATION_` | Evaluation / ground-truth dataset (not part of the RAG corpus) |

---

### Artificial Documents (`ART_`)

Designed with deliberate flaws to demonstrate RAG failure modes.

| File | Type | What it demonstrates |
|--------|--------|--------|
| `ART_product_catalog.pdf` | Internal policy | Portfolio scope, evidence-level policy, customer communication rules, open action items. The machine-readable product table lives in `product_overview.xlsx`. |
| `ART_internal_procurement_policy.pdf` | Internal policy | Evidence levels (A–D), EPD requirements, PFAS policy |
| `ART_supplier_brochure_tesa_ECO.pdf` | Supplier marketing | **Unverified claim**: 68% CO₂ reduction, no EPD |
| `ART_supplier_brochure_CPR_wood_pallet.pdf` | Supplier marketing | **Unverified LCA figures**; FSC claim without certificate |
| `ART_logylight_incomplete_datasheet.pdf` | Supplier datasheet | **Missing data**: all LCA fields "not yet available" |
| `ART_relicyc_logypal1_old_datasheet_2021.pdf` | Superseded document | **Temporal conflict**: 2021 figure vs. 2023 verified EPD |
| `ART_customer_inquiry_frische_felder.pdf` | Customer communication | CSRD request with open [ACTION] items |

---

### Environmental Product Declarations (`EPD_`)

Third-party verified EPDs. Naming: `EPD_{category}_{supplier}_{product}.pdf`

**Pallets:**

| File                                | Supplier      | Product (ID)              |
|-------------------------------------|---------------|---------------------------|
| `EPD_pallet_CPR_noe.pdf`            | CPR System    | Noé Pallet (32-100)       |
| `EPD_pallet_relicyc_logypal1.pdf`   | Relicyc       | Logypal 1 (32-103) — 2023 |
| `EPD_pallet_stabilplastik_ep08.pdf` | StabilPlastik | EP 08 (32-105)            |

**Cardboard Boxes:**

| File                                    | Supplier | Product (ID)                  |
|-----------------------------------------|----------|-------------------------------|
| `EPD_cardboard_redbox_cartonpallet.pdf` | Redbox   | Cartonpallet CMP (11-100)     |
| `EPD_cardboard_grupak_corrugated.pdf`   | Grupak   | Corrugated cardboard (11-101) |

**Tape:**

| File                              | Supplier | Product (ID)                  |
|-----------------------------------|----------|-------------------------------|
| `EPD_tape_IPG_hotmelt.pdf`        | IPG      | Hot Melt Tape (50-100)        |
| `EPD_tape_IPG_wateractivated.pdf` | IPG      | Water-Activated Tape (50-101) |

---

### Product Specifications & Datasheets (`SPEC_`)

Supplier-provided specs, datasheets, and article documents. Naming: `SPEC_{category}_{supplier}_{product}.pdf`

**Pallets:**

| File                                   | Supplier      | Product                      |
|----------------------------------------|---------------|------------------------------|
| `SPEC_pallet_CPR_plastic.pdf`          | CPR System    | Plastic Pallet (32-102)      |
| `SPEC_pallet_CPR_wood.pdf`             | CPR System    | Wooden Pallet (32-101)       |
| `SPEC_pallet_CPR_recycled_plastic.pdf` | CPR System    | Recycled plastic pallet PR12 |
| `SPEC_pallet_relicyc_logypal1.pdf`     | Relicyc       | Logypal 1: product info     |
| `SPEC_pallet_relicyc_1208MR.pdf`       | Relicyc       | Pallet 1208MR                |
| `SPEC_pallet_stabilplastik_ep08.pdf`   | StabilPlastik | EP 08: datasheet            |

**Tape:**

| File                                            | Supplier | Product                                     |
|-------------------------------------------------|----------|---------------------------------------------|
| `SPEC_tape_CST_synthetic_rubber.pdf`            | CST      | Synthetic rubber tape                       |
| `SPEC_tape_WAT_central_brand.pdf`               | WAT      | WAT tape: article document                 |
| `SPEC_tape_WAT_260.pdf`                         | WAT      | WAT 260: technical data sheet              |
| `SPEC_tape_tesa_tesapack58297.pdf`              | Tesa     | tesapack 58297 (German)                     |
| `SPEC_tape_tesa_sustainability_report_2024.pdf` | Tesa     | Company sustainability report 2024 (German) |

**Master data:**

| File                        | Content                                                                                                                                                            |
|-----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ART_product_overview.xlsx` | Machine-readable product table: all 12 products with IDs, suppliers, categories, and EPD status. Referenced by `ART_product_catalog.md` for the full product list. |

---

### Reference & Regulatory Documents (`REF_`)

| File                                         | Content                                                |
|----------------------------------------------|--------------------------------------------------------|
| `REF_ghg_protocol_product_lca.pdf`           | GHG Protocol: Product LCA standard                    |
| `REF_ghg_protocol_corporate_value_chain.pdf` | GHG Protocol: Scope 3 / value chain                   |
| `REF_ghg_protocol_corporate_standard.pdf`    | GHG Protocol: revised corporate standard              |
| `REF_eu_csrd.pdf`                            | EU Corporate Sustainability Reporting Directive (CSRD) |
| `REF_eu_climate_reporting_guidelines.pdf`    | EU guidelines on climate-related reporting             |
| `REF_ch_eu_sustainability_obligations.pdf`   | Swiss/EU sustainability obligations                    |
| `REF_iso14024_conformance_statement.pdf`     | ISO 14024 conformance statement                        |
| `REF_ecologo_catalogue.pdf`                  | ecoLogo certification catalogue                        |

---

### Evaluation Document

| File                            | Content                                                                                                                                              |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `EVALUATION_qa_ground_truth.md` | Sample questions with expected answers, required sources, and difficulty ratings. Use to manually verify RAG output. Not included in the RAG corpus. |

---

## Key Design Decisions in the Dataset

**Evidence quality varies deliberately.** Products span verified EPDs (Level A) to missing data (Level D). The RAG must distinguish these, not just retrieve numbers.

**Conflicts are intentional.** `ART_relicyc_logypal1_old_datasheet_2021.md` and `EPD_pallet_relicyc_logypal1.pdf` contain different GWP figures. The RAG should flag the conflict and prefer the more recent, verified source.

**Gaps are part of the answer.** For the LogyLight pallet and PFAS declarations, the correct answer is "we don't have this data." A RAG that fabricates an answer here fails.

**The portfolio boundary matters.** The product catalog defines which products exist. Queries about products outside the catalog (e.g., "Lara Pallet") should be answered as "not in our portfolio."

---

## LLM Backends

`BACKEND` must always be set explicitly, there is no default.

| Backend | Command | Notes |
|-----|-----|-----|
| Ollama | `BACKEND=ollama uv run python -m ki_kompass_rag.baseline_rag` | Requires `ollama serve` and `ollama pull mistral-nemo:12b` |
| OpenAI  | `BACKEND=openai uv run python -m ...` | Requires `OPENAI_API_KEY` env var or `/secrets/OPENAI_API_KEY` file |

Override the model: `MODEL=gpt-4o BACKEND=openai uv run python -m ...`

---

## Contributing

We welcome contributions: new feature notebooks, improvements to the toolkit library, additional documents in the dataset, or bug fixes. The steps below walk you through the full workflow from cloning to an approved pull request.

### 1. Clone the repository

```bash
git clone https://github.com/SwissDataScienceCenter/ki-kompass-conversational-toolkit.git
cd ki-kompass-conversational-toolkit
```

### 2. Set up your environment

```bash
# Create .venv and install all dependencies including the two editable packages
uv sync

# Install pre-commit hooks (runs formatters and linters before every commit)
uv run pre-commit install
```

### 3. Create a feature branch

Always work on a dedicated branch, never directly on `main`.

```bash
git checkout -b feature/my-descriptive-branch-name
```

Branch naming conventions:

- `feature/`: new functionality or notebooks
- `fix/`: bug fixes
- `docs/`: documentation-only changes
- `data/`: additions or corrections to the document corpus

### 4. Make your changes

A few guidelines depending on what you are changing:

**Notebooks (`backend/notebooks/`):**
- Keep cells focused and commented.
- Follow the naming convention `T{NN}_{short_topic}.ipynb` and keep the German version in sync.

**Toolkit library (`conversational-toolkit/`):**
- Add or update unit tests in the corresponding `tests/` directory.
- Run `pytest` before committing to confirm nothing is broken.
- Keep new classes and functions typed (the project uses `mypy`).

**Dataset (`data/`):**
- Follow the file-prefix convention: `EPD_`, `SPEC_`, `REF_`, `ART_`, or `EVALUATION_`.
- If adding a document that the RAG corpus should index, also add an entry to `README.md` under the appropriate `Dataset` subsection.
- If a new document introduces a deliberate flaw (conflict, missing data, unverified claim), describe it in the table.

### 5. Verify your changes locally

```bash
# Run the test suite
pytest
```

Fix any reported issues before continuing.

### 6. Commit your changes
Write concise commit messages (e.g. `fix chunker token-limit handling`, `add EPD for StabilPlastik EP08`).

```bash
git add path/to/changed/files
git commit -m "short description of what and why"
```

### 7. Push and open a pull request

```bash
git push -u origin feature/my-descriptive-branch-name
```

Then open a pull request on GitHub:

1. Go to the repository on GitHub and click **"Compare & pull request"** (the banner that appears after you push).
2. Set the **base branch** to `main`.
3. Write a clear PR description:
    - **What** does this PR change?
    - **Why** is the change needed or useful?
    - **How** was it tested (notebook run-through, `pytest`, manual check)?
    - If it closes an open issue, add `Closes #<issue-number>` in the description.
4. Request a review from at least one other contributor.

### 8. Respond to review feedback

Address comments by pushing additional commits to the same branch, do not open a new PR. Once all review threads are resolved and the CI checks pass, a maintainer will merge your PR into `main`.

---

## Development Reference

```bash
# Run tests
pytest

# Clear notebook outputs (replace with your notebook path)
uv run jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace backend/notebooks/english/T01_baseline_rag.ipynb
```
