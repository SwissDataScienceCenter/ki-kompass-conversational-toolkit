# KI-Kompass RAG

[Deutsche Version](README.de.md)

A retrieval-augmented generation (RAG) toolkit and a series of notebooks built on it. The toolkit is general-purpose; the notebooks and the chat app apply it to one example company, PrimePack AG. To use it for your own documents, see [Adapting to your own documents](backend/README.md#adapting-to-your-own-documents).

## Scenario

PrimePack AG sells pallets, cardboard boxes and tape from several suppliers. Its sustainability information is spread over EPDs, supplier brochures, datasheets, internal policies and regulations, and the quality of the evidence varies widely. Employees need answers to questions such as:

- *What does supplier A claim for product X, and can we trust it?*
- *Which suppliers meet our EPD requirement?*
- *Can we tell a customer that this tape is PFAS-free?*
- *Which evidence is missing before we can respond?*

A good assistant cites its sources, separates verified facts from supplier claims, points out gaps and conflicts, and declines to conclude when the evidence is insufficient.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the repository root:

```bash
uv sync
echo "OPENAI_API_KEY=sk-..." > .env.local
```

`uv sync` installs Python 3.13 and all dependencies into `.venv` from `uv.lock`, including the two workspace packages `conversational_toolkit` and `ki_kompass_rag`. The `ki_kompass_rag` package loads `.env.local` on import, so the key is available to the notebooks, the chat app and the command-line scripts. Prefix commands with `uv run` or activate `.venv`.

## Notebooks

```bash
uv run jupyter lab
```

Open `backend/notebooks/english/T01_baseline_rag.ipynb` and run the notebooks in order, each one top to bottom. `backend/notebooks/german/` has the same notebooks with German text; code, prompts and documents stay in English. The notebooks share one vector store, `backend/db/data_vs.db`, which is built from `data/` the first time it is needed.

| Notebook | Topic |
|---|---|
| `T01_baseline_rag` | The baseline RAG pipeline: chunk, embed, store, retrieve, generate |
| `T02_ingestion_chunking` | Document ingestion and chunking strategies |
| `T03_multimodal_rag` | Multimodal RAG: text and images |
| `T04_evaluation` | Evaluation and validation with RAGAS |
| `T05_synthetic_dataset` | Synthetic dataset generation |
| `T06_structured_outputs` | Reliable and structured outputs |
| `T07_advanced_retrieval` | Retrieval strategies: BM25, hybrid search, metadata filters |
| `T08_conversation` | Maintaining a conversation |
| `T09_query_techniques` | Query techniques: expansion and HyDE |
| `T10_tools` | Tools |
| `T11_llm_agents` | LLM agents |
| `T12_rag_tool` | RAG as a tool |
| `T13_rag_agent` | RAG agent |
| `T14_rag_subagent` | RAG as a subagent |

## Chat app

The chat app is a FastAPI backend (`ki_kompass_rag.main`, port 8080) and a Next.js frontend (port 3000). It needs Node.js and npm in addition to the setup above.

```bash
make build-db  # once: builds backend/db/vs_text and backend/db/vs_image
make run  # backend on http://localhost:8080
cd frontend && npm install && npm run dev  # second terminal; open http://localhost:3000
```

`make build-db` embeds image chunks with Qwen3-VL (a 2B-parameter model, slow on CPU). To build only the text store, run `uv run python -m ki_kompass_rag.build_app_db --skip-images`. The backend loads Qwen3-VL at startup either way, so the first start downloads the model.

## LLM backends

The chat app and the `baseline_rag` command line select the LLM with `BACKEND` (`openai` or `ollama`). The notebooks use OpenAI. Embeddings use `EMBEDDING_MODEL`, by default OpenAI `text-embedding-3-small`; any [sentence-transformers](https://huggingface.co/models?library=sentence-transformers) model name, e.g. `sentence-transformers/all-MiniLM-L6-v2`, runs locally instead. See [backend/README.md](backend/README.md#models) for the defaults.

To run the LLM locally, install [Ollama](https://ollama.com/download), pull a chat model from the [Ollama library](https://ollama.com/library) with `ollama pull <name>`, keep `ollama serve` running (the macOS app does this), and start the backend with `BACKEND=ollama MODEL=<name> make run`. Set `EMBEDDING_MODEL` to a local model as well to send nothing to OpenAI.

## Dataset

The documents in `data/` are the RAG corpus. The file prefix gives the document type:

| Prefix | Type |
|---|---|
| `ART_` | Artificial scenario document, written for the workshop |
| `EPD_` | Third-party verified Environmental Product Declaration |
| `SPEC_` | Supplier specification, datasheet or article document |
| `REF_` | Regulatory or reference document |
| `EVALUATION_` | Ground-truth questions and answers, excluded from the corpus |

The artificial documents contain deliberate flaws that test whether the assistant handles evidence correctly:

| File | Content | What it tests |
|---|---|---|
| `ART_product_catalog.pdf` | Product catalog | Portfolio scope, evidence-level policy, customer communication rules |
| `ART_product_overview.xlsx` | Product table with IDs, suppliers, categories and EPD status | Structured data |
| `ART_internal_procurement_policy.pdf` | Internal policy | Evidence levels A to D, EPD requirements, PFAS policy |
| `ART_supplier_brochure_tesa_ECO.pdf` | Supplier marketing | Unverified claim: 68% CO₂ reduction, no EPD |
| `ART_supplier_brochure_CPR_wood_pallet.pdf` | Supplier marketing | Unverified LCA figures; FSC claim without certificate |
| `ART_logylight_incomplete_datasheet.pdf` | Supplier datasheet | Missing data: all LCA fields "not yet available" |
| `ART_relicyc_logypal1_datasheet_2021.pdf` | Superseded datasheet | Conflict: its 2021 GWP figure differs from the 2023 verified EPD |
| `ART_response_inquiry_frische_felder.md` | Customer correspondence | CSRD request with open action items |


EPDs (`EPD_{category}_{supplier}_{product}.pdf`):

| File | Supplier | Product (ID) |
|---|---|---|
| `EPD_pallet_CPR_noe.pdf` | CPR System | Noé Pallet (32-100) |
| `EPD_pallet_relicyc_logypal1.pdf` | Relicyc | Logypal 1 (32-103), 2023 |
| `EPD_pallet_stabilplastik_ep08.pdf` | StabilPlastik | EP 08 (32-105) |
| `EPD_cardboard_redbox_cartonpallet.pdf` | Redbox | Cartonpallet CMP (11-100) |
| `EPD_cardboard_grupak_corrugated.pdf` | Grupak | Corrugated cardboard (11-101) |
| `EPD_tape_IPG_hotmelt.pdf` | IPG | Hot Melt Tape (50-100) |
| `EPD_tape_IPG_wateractivated.pdf` | IPG | Water-Activated Tape (50-101) |

Specifications and datasheets (`SPEC_{category}_{supplier}_{product}.pdf`):

| File | Supplier | Product |
|---|---|---|
| `SPEC_pallet_CPR_wood.pdf` | CPR System | Wooden Pallet (32-101) |
| `SPEC_pallet_CPR_plastic.pdf` | CPR System | Plastic Pallet (32-102) |
| `SPEC_pallet_CPR_recycled_plastic.pdf` | CPR System | Recycled plastic pallet PR12 |
| `SPEC_pallet_relicyc_logypal1.pdf` | Relicyc | Logypal 1 product information |
| `SPEC_pallet_relicyc_1208MR.pdf` | Relicyc | Pallet 1208MR |
| `SPEC_pallet_stabilplastik_ep08.pdf` | StabilPlastik | EP 08 datasheet |
| `SPEC_tape_CST_synthetic_rubber.pdf` | CST | Synthetic rubber tape |
| `SPEC_tape_WAT_central_brand.pdf` | WAT | WAT tape article document |
| `SPEC_tape_WAT_260.pdf` | WAT | WAT 260 technical datasheet |
| `SPEC_tape_tesa_tesapack58297.pdf` | Tesa | tesapack 58297 (German) |
| `SPEC_tape_tesa_sustainability_report_2024.pdf` | Tesa | Sustainability report 2024 (German) |

Reference documents:

| File | Content |
|---|---|
| `REF_ghg_protocol_product_lca.pdf` | GHG Protocol: product life cycle standard |
| `REF_ghg_protocol_corporate_value_chain.pdf` | GHG Protocol: Scope 3 / value chain |
| `REF_ghg_protocol_corporate_standard.pdf` | GHG Protocol: corporate standard |
| `REF_eu_csrd.pdf` | EU Corporate Sustainability Reporting Directive |
| `REF_eu_climate_reporting_guidelines.pdf` | EU guidelines on climate-related reporting |
| `REF_ch_eu_sustainability_obligations.pdf` | Swiss and EU sustainability obligations |
| `REF_iso14024_conformance_statement.pdf` | ISO 14024 conformance statement |
| `REF_ecologo_catalogue.pdf` | ECOLOGO certification catalogue |

## Repository layout

```
backend/                  ki_kompass_rag package: pipeline modules, chat app, notebooks
  notebooks/english/      T01 to T14
  notebooks/german/       T01 to T14, German text
  db/                     vector stores and app data, created at runtime (git-ignored)
conversational-toolkit/   conversational_toolkit library
frontend/                 Next.js chat UI
data/                     document corpus
docs/                     Renku guides for organisers and participants (German)
init.d/                   Renku session start script
tests/                    pytest suite for the toolkit and the backend
```

Package details: [backend/README.md](backend/README.md), [conversational-toolkit/README.md](conversational-toolkit/README.md), [frontend/README.md](frontend/README.md).

## Development

```bash
uv run pre-commit install  # ruff, mypy, nbstripout and file checks on every commit
make test  # uv run pytest -q
make lint  # ruff check, ruff format --check, mypy
```

CI (`.github/workflows/tests.yml`) runs ruff, mypy and pytest on Python 3.12 and 3.13. Pre-commit strips notebook outputs. Use branch prefixes `feature/`, `fix/`, `docs/` or `data/`. When you add a document to `data/`, follow the prefix convention and add it to the tables above.


## License

MIT, see [LICENSE](LICENSE).
