# KI-Kompass RAG

[English version](README.md)

Ein Toolkit für Retrieval-Augmented Generation (RAG) und eine Reihe von Notebooks, die darauf aufbauen. Das Toolkit ist an keinen bestimmten Anwendungsfall gebunden; die Notebooks und die Chat-App zeigen es am Beispiel der fiktiven Firma PrimePack AG. Wie man es für eigene Dokumente verwendet, steht unter [Eigene Dokumente verwenden](backend/README.de.md#eigene-dokumente-verwenden).

## Szenario

Die PrimePack AG verkauft Paletten, Kartons und Klebeband von mehreren Lieferanten. Ihre Nachhaltigkeitsinformationen sind über EPDs, Lieferantenbroschüren, Datenblätter, interne Richtlinien und Vorschriften verteilt, und die Qualität der Nachweise ist sehr unterschiedlich. Mitarbeitende brauchen Antworten auf Fragen wie:

- *Was gibt Lieferant A für Produkt X an, und können wir dem vertrauen?*
- *Welche Lieferanten erfüllen unsere EPD-Anforderung?*
- *Dürfen wir einem Kunden sagen, dass dieses Klebeband PFAS-frei ist?*
- *Welche Nachweise fehlen, bevor wir antworten können?*

Ein guter Assistent nennt seine Quellen, trennt verifizierte Fakten von Lieferantenangaben, weist auf Lücken und Widersprüche hin und zieht keine Schlüsse, wenn die Nachweise nicht ausreichen.

## Setup

[uv](https://docs.astral.sh/uv/getting-started/installation/) installieren, dann im Hauptverzeichnis des Repos:

```bash
uv sync
echo "OPENAI_API_KEY=sk-..." > .env.local
```

`uv sync` installiert Python 3.13 und alle Abhängigkeiten aus `uv.lock` in `.venv`, einschliesslich der beiden Pakete `conversational_toolkit` und `ki_kompass_rag`. `ki_kompass_rag` lädt `.env.local` beim Import, der Key steht also den Notebooks, der Chat-App und den Kommandozeilen-Skripten zur Verfügung. Befehle mit `uv run` ausführen oder `.venv` aktivieren.

## Notebooks

Im Workshop laufen die Notebooks auf Renku, ohne lokale Installation: siehe [Renku im KI-Kompass](docs/renku_teilnehmende.md).

```bash
uv run jupyter lab
```

`backend/notebooks/german/T01_baseline_rag.ipynb` öffnen und die Notebooks der Reihe nach ausführen, jedes von oben nach unten. Alle Notebooks verwenden denselben Vector Store, `backend/db/data_vs.db`. Er wird beim ersten Ausführen automatisch aus den Dokumenten in `data/` erstellt.

Die Erklärungen sind auf Deutsch. Code, Prompts und die Dokumente in `data/` bleiben auf Englisch: Die Dokumente sind englisch, und Fragen in derselben Sprache liefern meist bessere Treffer; die Prompts sind auf die Dokumente und das Testset abgestimmt. Zu jedem Prompt steht im Text eine deutsche Übersetzung. Die englischen Notebooks liegen in `backend/notebooks/english/`.

| Notebook | Thema |
|---|---|
| `T01_baseline_rag` | Die Baseline-RAG-Pipeline: chunken, embedden, speichern, suchen, generieren |
| `T02_ingestion_chunking` | Ingestion von Dokumenten und Chunking-Strategien |
| `T03_multimodal_rag` | Multimodales RAG: Text und Bilder |
| `T04_evaluation` | Evaluation und Validierung mit RAGAS |
| `T05_synthetic_dataset` | Synthetische Testdaten erzeugen |
| `T06_structured_outputs` | Zuverlässige und strukturierte Ausgaben |
| `T07_advanced_retrieval` | Retrieval-Strategien: BM25, hybride Suche, Metadatenfilter |
| `T08_conversation` | Eine Konversation führen |
| `T09_query_techniques` | Query-Techniken: Expansion und HyDE |
| `T10_tools` | Tools |
| `T11_llm_agents` | LLM-Agenten |
| `T12_rag_tool` | RAG als Tool |
| `T13_rag_agent` | RAG-Agent |
| `T14_rag_subagent` | RAG als Subagent |

## Chat-App und lokale Modelle

Die Chat-App (FastAPI-Backend und Next.js-Frontend), lokale LLMs über Ollama und lokale Embedding-Modelle sind im [englischen README](README.md#chat-app) beschrieben.

## Datensatz

Die Dokumente in `data/` bilden den Korpus für das RAG. Das Präfix im Dateinamen gibt den Dokumenttyp an:

| Präfix | Typ |
|---|---|
| `ART_` | Fiktives Szenario-Dokument, für den Workshop geschrieben |
| `EPD_` | Extern verifizierte Umweltproduktdeklaration (Environmental Product Declaration) |
| `SPEC_` | Spezifikation, Datenblatt oder Artikeldokument eines Lieferanten |
| `REF_` | Regulatorisches Dokument oder Referenzdokument |
| `EVALUATION_` | Testfragen mit Referenzantworten, nicht Teil des Korpus |

Die fiktiven Dokumente enthalten absichtliche Mängel, die prüfen, ob der Assistent richtig mit Nachweisen umgeht:

| Datei | Inhalt | Was geprüft wird |
|---|---|---|
| `ART_product_catalog.pdf` | Produktkatalog | Sortiment, Richtlinie zu Nachweisstufen, Regeln für die Kundenkommunikation |
| `ART_product_overview.xlsx` | Produkttabelle mit IDs, Lieferanten, Kategorien und EPD-Status | Strukturierte Daten |
| `ART_internal_procurement_policy.pdf` | Interne Richtlinie | Nachweisstufen A bis D, EPD-Anforderungen, PFAS-Richtlinie |
| `ART_supplier_brochure_tesa_ECO.pdf` | Lieferantenwerbung | Nicht verifizierte Angabe: 68 % CO₂-Reduktion, keine EPD |
| `ART_supplier_brochure_CPR_wood_pallet.pdf` | Lieferantenwerbung | Nicht verifizierte LCA-Werte; FSC-Angabe ohne Zertifikat |
| `ART_logylight_incomplete_datasheet.pdf` | Datenblatt eines Lieferanten | Fehlende Daten: alle LCA-Felder "not yet available" |
| `ART_relicyc_logypal1_datasheet_2021.pdf` | Veraltetes Datenblatt | Widerspruch: GWP-Wert von 2021 weicht von der verifizierten EPD von 2023 ab |
| `ART_response_inquiry_frische_felder.md` | Kundenkorrespondenz | CSRD-Anfrage mit offenen Punkten |

Die Listen der EPDs, Datenblätter und Referenzdokumente stehen im [englischen README](README.md#dataset).

## Lizenz

MIT, siehe [LICENSE](LICENSE).
