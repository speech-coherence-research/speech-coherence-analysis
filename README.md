# Discourse Coherence Metrics
This repository contains the code used to calculate the discourse coherence measures reported in the associated study.
The files in `demo_data/` are artificial examples provided only to demonstrate the workflow. They are not part of the research dataset. No participant data are included.

## Installation and use
Python 3.10 is recommended.

```bash
pip install -r requirements.txt
python calculate_coherence_metrics.py
```
The initial installation or execution may require internet access to download required packages and language models.

## Input
Input files should follow this structure:
```text
demo_data/
└── Group_1/
    └── Task_01/
        ├── Reference.txt
        ├── P01.txt
        └── P02.txt
```
The input and output locations can be changed near the beginning of `calculate_coherence_metrics.py`:

```python
INPUT_DIR = "demo_data"
OUTPUT_FILE = "results/coherence_metrics_results.xlsx"
```

## Output
The script generates an Excel file containing the calculated discourse coherence measures. Definitions of the measures are provided in `CODEBOOK.md`.

## Notes
Results may vary across operating systems, dependency versions, language-model versions, and preprocessing procedures. The code was developed for English transcripts. Users should evaluate its suitability for their own data.
Third-party software and language models remain subject to their respective licenses.