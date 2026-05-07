# Cypriot Greek POS Tagger

This project trains and evaluates part-of-speech taggers for Cypriot Greek using a custom annotated dataset

Two traditional NLP models are compared:

- Brill's tagger
- Hidden Markov Model (HMM) tagger

Both models are evaluated with:

- Simplified POS tagset
- Detailed POS tagset
- 5-fold cross-validation
- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Per-tag F1 analysis
- Qualitative error inspection

## Research Background

This project is based on my undergraduate thesis on part-of-speech tagging for Cypriot Greek.

The thesis paper is not included in this repository. If you'd be interested in reading it, please contact me through GitHub.

## Dataset

The full annotated dataset is not included in this repository because it is part of a currently unpublished research project

To run the full experiments with the original dataset, please contact me through GitHub to request access

The expected dataset columns are:

- Sentence ID
- Word
- POS Tag

The dataset is loaded from Google Sheets using a local Google service account credentials file

The credentials file is not included in this repository because it contains private credentials

Expected local credentials filename:

```text
cypriot_pos_credentials.json
```

Alternatively, the credentials path can be set with:

```bash
export CYPRIOT_CREDS="/path/to/cypriot_pos_credentials.json"
```

## Method

The models are evaluated using 5-fold cross-validation at the sentence level

For Brill's tagger, a unigram tagger is used as the initial tagger with a regular-expression tagger and default noun tagger as backoff components

For the HMM tagger, supervised training is used with Lidstone smoothing to reduce zero-probability issues in sparse transition and emission estimates

Macro F1 is reported because the POS tag distribution is imbalanced and frequent tags can otherwise dominate the evaluation

## Results

The following table summarizes the 5-fold cross-validation results on the manually created Cypriot Greek dataset

| Model | Tagset | Accuracy | Macro F1 |
|---|---|---:|---:|
| Brill tagger | Simplified | 70.18 | 66.29 |
| Brill tagger | Detailed | 66.62 | 41.65 |
| HMM tagger | Simplified | 64.59 | 56.18 |
| HMM tagger | Detailed | 56.12 | 31.92 |

## Installation

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

## Running the taggers

Run the scripts from the project root:

```bash
python scripts/train_brill.py
python scripts/train_brilldetailed.py
python scripts/train_hmm.py
python scripts/train_hmm.detailed.py
```

Depending on your system, you may need to use `python3` instead:

```bash
python3 scripts/train_brill.py
```

## Outputs

The scripts print cross-validation results in the terminal and generate output files for analysis

Example outputs include:

```text
brill_per_tag_f1_simplified_cv.png
brill_per_tag_f1_detailed_cv.png
hmm_per_tag_f1_simplified_cv.png
hmm_per_tag_f1_detailed_cv.png
brill_qual_errors_simplified.tsv
brill_qual_errors_detailed.tsv
hmm_qual_errors_simplified.tsv
hmm_qual_errors_detailed.tsv
```

These files are generated locally when the scripts are run
