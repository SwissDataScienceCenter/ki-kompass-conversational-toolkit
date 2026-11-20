"""Hand-written evaluation queries with ground-truth answers, checked against the documents in data/."""

EVALUATION_QUERIES: list[dict] = [
    {
        "query": "Does PrimePack AG offer a product called the Lara Pallet?",
        "ground_truth_answer": (
            "No. No product called Lara Pallet appears in any PrimePack AG document. The active portfolio is listed in product_overview.xlsx, and its pallets are: Noé pallet (32-100, CPR System), Wooden pallet (32-101, CPR System), Plastic pallet (32-102, CPR System), Logypal 1 (32-103, Relicyc), LogyLight (32-104, Relicyc), and Plastic pallet EP 08 (32-105, StabilPlastik)."
        ),
        "expected_keywords": ["not", "no", "portfolio"],
        "category": "portfolio_scope",
        "difficulty": "easy",
    },
    {
        "query": "Which products in the portfolio have a third-party verified EPD?",
        "ground_truth_answer": (
            "Products with an EPD: 50-100 (IPG Pressure-Sensitive Hot Melt Carton Sealing Tape), 50-101 (IPG Water-Activated Tape), 32-100 (Noé pallet, CPR System), 32-103 (Logypal 1, Relicyc), 32-105 (Plastic pallet EP 08, StabilPlastik), 11-100 (Cartonpallet CMP Roserio, redbox), 11-101 (Corrugated cardboard packaging, Grupak). Products without an EPD: 50-102 (tesapack ECO & ULTRA STRONG ecoLogo), 32-101, 32-102, 32-104."
        ),
        "expected_keywords": ["50-100", "50-101", "32-100", "32-103", "32-105"],
        "category": "portfolio_scope",
        "difficulty": "easy",
    },
    {
        "query": "Can the 68% CO2 reduction claim for tesapack ECO (product 50-102) be included in a customer sustainability response?",
        "ground_truth_answer": (
            "Not as a verified fact. The 68% CO2 reduction per roll comes from an internal comparative assessment by Tesa SE (2019 baseline) and is not verified by an EPD or third-party LCA. Under PrimePack AG's supplier policy it is at best Level B evidence and may only be cited with the caveat 'self-declared, not independently verified'; GWP figures shared with customers must come from a verified EPD. The carbon neutrality goal (end of 2025 in the text, 2026 in the brochure's timeline) is a target, not a current status."
        ),
        "expected_keywords": ["not", "EPD", "internal"],
        "category": "claim_verification",
        "difficulty": "hard",
    },
    {
        "query": "What verified environmental data is available for the LogyLight pallet (product 32-104)?",
        "ground_truth_answer": (
            "None. The LogyLight sustainability information sheet (November 2024) lists GWP and all other impact categories as 'Data not yet available' (LCA in preparation). An ISO 14044 LCA has been commissioned (study reference REL-LCA-2024-07); the sheet expects a third-party verified EPD by Q2 2026, while the supplier policy's January 2025 status table says Q2 2025. The 75% post-consumer recycled content and the 100% recyclability are self-declared, without third-party audit. LogyLight must not be given any GWP or environmental figure."
        ),
        "expected_keywords": ["not yet available", "LCA", "EPD"],
        "category": "missing_data",
        "difficulty": "easy",
    },
    {
        "query": "Which GWP source should be used for Relicyc Logypal 1: the 2021 datasheet or the 2023 EPD?",
        "ground_truth_answer": (
            "The 2023 third-party verified EPD (EPD International registration S-P-09135, published 2023-06-26). The June 2021 datasheet reports 4.1 kg CO2e per pallet (50-trip lifetime) from Relicyc's internal 2020 LCA, which it states was not independently verified. PrimePack AG's policy says to flag conflicting figures, state both values, and prefer the more recent third-party verified source; the 4.1 kg figure must not be cited as verified."
        ),
        "expected_keywords": ["EPD", "2023", "verified"],
        "category": "source_conflict",
        "difficulty": "hard",
    },
    {
        "query": "Are any tape products confirmed to be PFAS-free?",
        "ground_truth_answer": (
            "No. As of January 2025, PFAS declarations are outstanding for all tape suppliers (IPG and Tesa SE), and PrimePack AG must not make PFAS-free claims until they are received. Tesa's 'hot-melt, free of intentionally added solvents' adhesive is not a PFAS declaration."
        ),
        "expected_keywords": ["no", "declaration"],
        "category": "missing_data",
        "difficulty": "medium",
    },
    {
        "query": "Which suppliers are not yet compliant with the EPD requirement by end of 2025?",
        "ground_truth_answer": (
            "As of January 2025, Tesa SE (50-102, no EPD, carbon claims unverified) and CPR System for 32-101 and 32-102 (no EPD, internal calculation only) are non-compliant. Relicyc's LogyLight (32-104) is in progress: LCA commissioned, EPD pending."
        ),
        "expected_keywords": ["Tesa", "CPR"],
        "category": "policy",
        "difficulty": "easy",
    },
]
