# Amazon ML Challenge 2026: Business Entity Resolution

Reproducible machine learning pipeline for the Amazon ML Challenge 2026 Business Entity Resolution challenge.

---

## 📌 Problem Overview
- **Reference Dataset (`Source 1`)**: Source of truth containing business entity records.
- **Target Datasets (`Source 2` and `Source 3`)**: Candidate sources containing entities to be linked to Source 1.
- **Objective**: For every Source 1 entity, identify zero, one, or multiple matching entities from Source 2 and Source 3.

---

## 📁 Project & Dataset Structure

```
├── configs/
│   └── config.yaml             # Central configuration (paths, schema, parameters)
├── data/
│   ├── train/
│   │   ├── train_source1.tsv       # 2,206,821 records (Master reference entities)
│   │   ├── train_source2.tsv       # 5,034,616 records (Candidate source 2)
│   │   ├── train_source3.tsv       # 5,285,603 records (Candidate source 3)
│   │   └── train_ground_truth.tsv  # 2,206,821 records (S1 -> S2, S3 match links)
│   └── test/
│       ├── test_source1.tsv        # 1,732,544 records
│       ├── test_source2.tsv        # 4,887,273 records
│       └── test_source3.tsv        # 5,082,316 records
├── docs/                       # Project documentation
├── experiments/                # Experiment logs and metrics
├── models/                     # Model weights and artifacts
├── notebooks/
│   └── 01_eda.ipynb            # Comprehensive exploratory data analysis
├── output/                     # Prediction submissions
├── src/
│   ├── __init__.py
│   └── data_loader.py          # Memory-efficient streaming DataLoader
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 📊 Dataset Columns & Schemas

### Entity Files (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`):
| Column Name | Type | Description | Missing Rate |
| :--- | :--- | :--- | :--- |
| `entity_id` | String | Unique ID prefixed by source (`S1-`, `S2-`, `S3-`) | 0% |
| `business_name` | String | Commercial entity / brand / business name | 0% |
| `business_address` | String | Street, locality, city, state, postal address | ~2.6% - 3.4% in S2 & S3; 0% in S1 |
| `country` | String | Country string identifier (open-set) | 0% |

### Ground Truth File (`train_ground_truth.tsv`):
| Column Name | Type | Description |
| :--- | :--- | :--- |
| `source1_entity_id` | String | Reference entity ID (`S1-...`) |
| `matched_entity_ids` | String | Comma-separated list of matching entity IDs from S2 and S3 |

---

## 🔍 Key EDA Findings & Statistics

### 1. Volume & Uniqueness
- **Total Records Analyzed**: 26,436,014 records across 7 TSV files.
- **Entity ID Uniqueness**: 100% unique within each file (0 duplicate IDs).
- **Source 1 Ground Truth Coverage**: Exactly 2,206,821 rows, perfectly 1-to-1 matching `train_source1.tsv`.

### 2. Match Dynamics (Training Ground Truth)
- **Total Ground Truth Matches**: 7,638,365 matched pairs.
  - **S1 → S2 Matches**: 3,693,619 (48.36%)
  - **S1 → S3 Matches**: 3,944,746 (51.64%)
- **Zero-Match (Singleton) S1 Entities**: 123,247 (5.58%)
- **Single-Match S1 Entities**: 119,157 (5.40%)
- **Multiple Matches (>1 Match)**: 1,964,417 (**89.02%**)
- **Mean Matches per S1 Entity**: 3.46 (Median: 3, Max: 11).

### 3. Open-Set Country Distribution
- **Training Set**: 2 countries — `US` (~60%) and `India` (~40%).
- **Test Set**: 3 countries — `India` (~47%), `US` (~38%), and **`France` (~15%)** (a new country unseen in training).
- **Takeaway**: Country normalization and blocking rules must remain open-set and generalizable across international formats.

### 4. Text Lengths & Character Profiles
- **Business Names**: Mean length 24–25 characters, range 2–123 chars. High rate of shared non-unique names (e.g., medical/retail chains).
- **Business Addresses**: Mean length 46–57 characters, range up to 269 chars. ~3% missing in target candidate sources S2 and S3.

---

## ⚠️ Important Constraints & Data-Quality Observations

1. **Open-Set Country**: Never hardcode country filters to US/India only.
2. **Missing Addresses in Candidates**: Address similarity cannot be the sole matching signal; name similarity and robust token overlap must handle records where address is empty.
3. **Multi-Match Cardinality**: 89% of S1 entities link to multiple entities across both S2 and S3 simultaneously. The matching pipeline must produce variable-length multi-label links.
4. **Encoding & Special Characters**: Multiple languages (including Devanagari Hindi and French accented characters) require UTF-8 handling and clean unicode normalization.
5. **Memory Scalability**: Total data volume exceeds 26 million rows; chunked/streaming readers and efficient indexing (inverted index / candidate blocking) are mandatory to avoid OOM.

---

## 🛠️ Setup & Usage

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```
