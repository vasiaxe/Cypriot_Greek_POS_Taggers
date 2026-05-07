import random
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from nltk.tag import UnigramTagger, RegexpTagger, DefaultTagger
from nltk.tag.brill import Pos, Word, Template
from nltk.tag.brill_trainer import BrillTaggerTrainer

from sklearn.model_selection import KFold
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from preprocessing.prepare_data import load_data_from_gsheet, group_sentences

# Fix random seeds so the shuffled cross-validation split is reproducible
random.seed(42)
np.random.seed(42)

# Load the annotated dataset with the original detailed POS labels
df = load_data_from_gsheet("cypriot_pos_credentials.json", "Data", simplify=False)

# Convert token-level spreadsheet rows into sentence-level lists of word-tag pairs
sentences = group_sentences(df)
if not sentences:
    raise ValueError("No sentences found after grouping.")

print(f"Total sentences (detailed): {len(sentences)}")

# Backoff chain uses regex rules first and defaults to Noun (lexical) if no pattern matches
default_tagger = DefaultTagger("Noun (lexical)")

# Handwritten Cypriot Greek form-based patterns mapped to detailed POS tags
patterns = [
    (r".*εν$", "Verb (lexical)"),
    (r".*ος$", "Noun (lexical)"),
    (r".*α$", "Noun (lexical)"),
    (r".*ως$", "Lexical Adverb"),
    (r".*ικά$", "Lexical Adverb"),
    (r"^τες$", "Determiner: Article"),
    (r"^έ.*", "Verb (lexical)"),
    (r".*ει$", "Verb (lexical)"),
]
regexp_tagger = RegexpTagger(patterns, backoff=default_tagger)

# Brill templates define which contextual features can be used to learn correction rules
templates = [
    Template(Pos([-2])), Template(Pos([-1])), Template(Pos([1])), Template(Pos([2])),
    Template(Pos([-1, 1])), Template(Word([-1])), Template(Word([1]))
]

# Use 5-fold cross-validation because the dataset is small
# Each sentence is used for testing once while the model is trained on the other folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_accuracies = []
fold_precisions = []
fold_recalls = []
fold_f1s = []

# Store all cross-validated predictions for per-tag analysis
cv_y_true = []
cv_y_pred = []

# Store individual tagging mistakes for qualitative error analysis
qual_errors = []

print("\nBrill – 5-fold CV on DETAILED tagset")

for fold, (train_idx, test_idx) in enumerate(kf.split(sentences), start=1):
    print(f"\n=== Fold {fold}/5 ===")

    train_sents = [sentences[i] for i in train_idx]
    test_sents = [sentences[i] for i in test_idx]

    # Train the initial unigram tagger only on the current training fold
    # Brill then learns transformation rules that correct the initial tagging
    base = UnigramTagger(train_sents, backoff=regexp_tagger)
    trainer = BrillTaggerTrainer(initial_tagger=base, templates=templates)
    brill = trainer.train(train_sents, max_rules=200)

    y_true_f, y_pred_f = [], []

    for sent in test_sents:
        words = [w for w, _ in sent]
        gold = [t.strip() for _, t in sent]
        pred = [t.strip() for _, t in brill.tag(words)]

        y_true_f.extend(gold)
        y_pred_f.extend(pred)

        # Save misclassified tokens with immediate context for manual inspection
        sent_str = " ".join(words)
        for i, (w, g, p) in enumerate(zip(words, gold, pred)):
            if g != p:
                left_ctx = words[i - 1] if i > 0 else "<BOS>"
                right_ctx = words[i + 1] if i < len(words) - 1 else "<EOS>"
                qual_errors.append((fold, w, g, p, left_ctx, right_ctx, sent_str))

    if not y_true_f:
        continue

    cv_y_true.extend(y_true_f)
    cv_y_pred.extend(y_pred_f)

    # Macro-F1 gives each POS tag equal weight instead of letting frequent tags dominate
    acc = accuracy_score(y_true_f, y_pred_f)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_f, y_pred_f, average="macro", zero_division=0
    )

    fold_accuracies.append(acc)
    fold_precisions.append(prec)
    fold_recalls.append(rec)
    fold_f1s.append(f1)

    print(f"Fold {fold} accuracy: {acc:.2%} | macro-F1: {f1:.2%}")

# Report mean scores and fold-to-fold variation across the 5 folds
print("\nBrill (DETAILED) CV summary:")
print(f"Accuracy:  {np.mean(fold_accuracies):.2%} (±{np.std(fold_accuracies):.2%})")
print(f"Precision: {np.mean(fold_precisions):.2%}")
print(f"Recall:    {np.mean(fold_recalls):.2%}")
print(f"F1:        {np.mean(fold_f1s):.2%} (±{np.std(fold_f1s):.2%})")

# Print a small sample of errors for quick terminal inspection
print("\nSample qualitative errors (Brill, detailed):")
print(f"{'Fold':<5} {'Word':<14} {'Gold':<25} {'Pred':<25} {'Left':<14} {'Right':<14}")
print("-" * 105)

for e in qual_errors[:20]:
    fold, w, g, p, l, r, _sent = e
    print(f"{fold:<5} {w:<14} {g:<25} {p:<25} {l:<14} {r:<14}")

# Save the full error list for qualitative analysis in the report
with open("brill_qual_errors_detailed.tsv", "w", encoding="utf-8") as f:
    f.write("fold\tword\tgold\tpred\tleft_ctx\tright_ctx\tsentence\n")
    for fold, w, g, p, l, r, sent_str in qual_errors:
        f.write(f"{fold}\t{w}\t{g}\t{p}\t{l}\t{r}\t{sent_str}\n")

print("\nSaved qualitative errors to: brill_qual_errors_detailed.tsv")

# Clean tag strings before per-tag analysis to avoid duplicate labels caused by whitespace
cv_y_true = [t.strip() for t in cv_y_true]
cv_y_pred = [t.strip() for t in cv_y_pred]

labels = sorted(set(cv_y_true) | set(cv_y_pred))

# Compute per-tag precision recall F1 and support from all cross-validated predictions
precisions, recalls, f1s, support = precision_recall_fscore_support(
    cv_y_true, cv_y_pred, labels=labels, zero_division=0
)

print("\nPer-tag F1 (Brill, detailed tagset):")
print(f"{'Tag':<30} {'F1':>6} {'Count':>7}")
print("-" * 50)
for tag, f1, n in zip(labels, f1s, support):
    print(f"{tag:<30} {f1:6.2f} {n:7d}")

# Use a confusion matrix to calculate how often each gold tag was misclassified
cm = confusion_matrix(cv_y_true, cv_y_pred, labels=labels)
correct = cm.diagonal()
total = cm.sum(axis=1)
error_rates = 1 - (correct / total)

print("\nPer-tag error rates (Brill, detailed tagset):")
print(f"{'Tag':<30} {'Count':>7} {'Correct':>8} {'Error%':>8}")
print("-" * 60)
for tag, n, c, e in zip(labels, total, correct, error_rates):
    if n == 0:
        continue
    print(f"{tag:<30} {n:7d} {c:8d} {100*e:8.2f}")

# Save a per-tag F1 bar chart for inclusion in the report
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x, f1s)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=90)
ax.set_ylim(0, 1.0)
ax.set_ylabel("F1-score")
ax.set_title("Brill – Per-tag F1 (5-fold CV, detailed)")
plt.tight_layout()
plt.savefig("brill_per_tag_f1_detailed_cv.png", dpi=300)
plt.close(fig)

print("\nSaved plot to: brill_per_tag_f1_detailed_cv.png")