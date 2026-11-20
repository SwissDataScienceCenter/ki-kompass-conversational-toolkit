# RAG Evaluation: Ground Truth Question-Answer Pairs

*Five hand-crafted test cases for evaluating the PrimePack AG RAG assistant.*
*A correct answer does not need to match word-for-word, but must contain the key facts and correctly represent uncertainty where uncertainty exists.*

---

## Q1: portfolio_scope

**Question:** Does PrimePack AG offer a product called the "Lara Pallet"?

**Expected answer:** No. No product called Lara Pallet appears in any PrimePack AG document. The product catalog states that all active products are listed in `product_overview.xlsx`, and that list contains no Lara Pallet. The active pallets are: Noé pallet (32-100, CPR System), Wooden pallet (32-101, CPR System), Plastic pallet (32-102, CPR System), Logypal 1 (32-103, Relicyc), LogyLight (32-104, Relicyc), and Plastic pallet EP 08 (32-105, StabilPlastik). If a customer asks about the Lara Pallet, the correct response is to refer them to the current product overview.

**Primary sources:**
- `data/ART_product_overview.xlsx`
- `data/ART_product_catalog.pdf`

**Failure mode to watch:** The system invents a description for the Lara Pallet — e.g. by extrapolating from other pallet documents — instead of stating the product does not exist.

---

## Q2: claim_verification

**Question:** Can the 68% CO₂ reduction claim for the tesapack ECO & ULTRA STRONG ecoLogo (product 50-102) be included in a customer sustainability response?

**Expected answer:** No — not as a stated fact. The 68% reduction per roll (versus Tesa's 2019 conventional baseline) comes from an internal comparative assessment by Tesa SE and has not been verified through an EPD or third-party LCA. Under PrimePack AG's supplier policy this is at best Level B evidence (Level C claims may not be communicated at all). All GWP / carbon footprint figures shared with customers must come from Level A evidence (a verified EPD). The claim may only be mentioned with the caveat *"self-declared, not independently verified."* The carbon neutrality goal (end of 2025 in the brochure text, 2026 in its timeline) is a forward-looking target, not a current status, and must be labelled as such.

**Primary sources:**
- `data/ART_supplier_brochure_tesa_ECO.pdf`
- `data/ART_internal_procurement_policy.pdf`

**Failure mode to watch:** The system presents the 68% figure as a verified or directly shareable fact, or describes carbon neutrality as achieved rather than as a future target.

---

## Q3: missing_data

**Question:** What verified environmental data is available for the LogyLight pallet (product 32-104)?

**Expected answer:** None. The LogyLight sustainability information sheet (November 2024) lists GWP and all other impact categories as *"Data not yet available"* (LCA in preparation). A full lifecycle assessment (ISO 14044) has been commissioned (study reference: REL-LCA-2024-07). The sheet expects a third-party verified EPD by Q2 2026, while the supplier policy's January 2025 status table says Q2 2025; either way no verified figures exist. The 75% post-consumer recycled content and the "100% theoretically recyclable" statement are Relicyc self-declarations without independent audit. The LogyLight must not be included in customer-facing environmental comparisons until a verified EPD is published.

**Primary sources:**
- `data/ART_logylight_incomplete_datasheet.pdf`
- `data/ART_internal_procurement_policy.pdf`

**Failure mode to watch:** The system extrapolates environmental figures from the Logypal 1 EPD, or presents the "in preparation" status as if usable data already exists.

---

## Q4: missing_data

**Question:** Are any of PrimePack AG's tape products confirmed to be PFAS-free?

**Expected answer:** No. As of January 2025, PFAS declarations are outstanding for all tape suppliers (IPG and Tesa SE); receiving them is an open action item in the product catalog, which states: *"Do not make PFAS-free claims until declarations are received."* The supplier policy requires every tape supplier to provide an explicit PFAS declaration per product. The tesa brochure mentions a "hot-melt, free of intentionally added solvents" adhesive system, but this does not constitute a PFAS declaration. No tape product may be described as PFAS-free until explicit supplier declarations are received and reviewed.

**Primary sources:**
- `data/ART_product_catalog.pdf`
- `data/ART_internal_procurement_policy.pdf`
- `data/ART_supplier_brochure_tesa_ECO.pdf`
- `data/ART_response_inquiry_frische_felder.md`

**Failure mode to watch:** The system infers PFAS-free status from the phrase "free of intentionally added solvents" in the tesa brochure, or from the absence of any mention of PFAS in a technical data sheet.

---

## Q5: source_conflict

**Question:** Which GWP figure should be cited for the Relicyc Logypal 1 pallet (product 32-103), and why?

**Expected answer:** The authoritative figure is in the third-party verified EPD published on 2023-06-26 (EPD International registration S-P-09135, `EPD_pallet_relicyc_logypal1.pdf`). The June 2021 datasheet (`ART_relicyc_logypal1_datasheet_2021.pdf`) reports a GWP of 4.1 kg CO₂e per pallet (50-trip lifetime) from Relicyc's internal 2020 LCA (SimaPro, ecoinvent 3.6, mass allocation) and states that *"these figures were not independently verified at time of publication"*; at that time an EPD was only in planning. The 4.1 kg CO₂e figure must not be cited as the current or verified GWP. When two sources conflict, PrimePack AG's policy requires flagging the conflict, stating both values, and preferring the more recent third-party verified source.

**Primary sources:**
- `data/EPD_pallet_relicyc_logypal1.pdf`
- `data/ART_relicyc_logypal1_datasheet_2021.pdf`
- `data/ART_product_catalog.pdf` (conflicting sources rule)

**Failure mode to watch:** The system cites 4.1 kg CO₂e as the current or verified GWP, without noting that it is an unverified 2021 figure, or without preferring the 2023 EPD.

---

## Scoring Guide

| Score | Criteria |
|---|---|
| Correct | Key facts present; uncertainty correctly expressed; no fabricated data; conflicting sources handled |
| Partial | Main facts correct but evidence level, uncertainty, or source conflict not addressed |
| Incorrect | Wrong facts, fabricated data, or unverified claims presented as verified |
| Hallucination | Describes entities or data absent from or contradicted by all source documents |

**Red flags -> automatic incorrect:**
- Describes the Lara Pallet as if it exists in the portfolio
- Presents the tesa 68% CO₂ figure as independently verified
- States carbon neutrality for tesapack ECO as a current achieved status
- Confirms PFAS-free status for any tape product (no evidence exists)
- Cites 4.1 kg CO₂e as the current verified GWP for Logypal 1
- Provides any GWP or environmental figure for LogyLight
