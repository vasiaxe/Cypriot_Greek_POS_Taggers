import os
from pathlib import Path

import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials


# Tag simplification function
def simplify_tag(tag: str) -> str:
    tag = str(tag).strip().lower()

    if tag == "pronoun": return "PRON"
    if "adjective" in tag: return "ADJ"
    if "adverb" in tag: return "ADV"
    if tag == "interjection": return "INTJ"
    if tag == "gerund": return "NOUN"
    if "noun" in tag: return "NOUN"
    if "determiner" in tag: return "DET"
    if "particle" in tag: return "PART"
    if tag == "preposition": return "ADP"
    #if "modal verb" in tag: return "VERB"
    if "verb" in tag: return "VERB"
    if "auxiliary" in tag or "copula" in tag: return "AUX"
    if "coordinating conjunction" in tag: return "CONJ"
    if "subordinating conjunction" in tag or "complimentiser" in tag: return "SCONJ"
    return "X"


# Resolve credentials path once, so callers can just pass a filename.
def _resolve_creds_path(json_key_path: str) -> Path:
    """
    Resolve a Google service-account JSON path robustly.

    Resolution order:
      1) CYPRIOT_CREDS env var (if set)
      2) The given path (absolute or relative to CWD)
      3) Project root (based on this file) + given path
      4) Project root / resources / <filename>

    Raises FileNotFoundError with all attempted paths if nothing is found.
    """
    tried = []

    # 1) Environment variable override
    env_path = os.getenv("CYPRIOT_CREDS")
    if env_path:
        p = Path(env_path).expanduser()
        tried.append(str(p))
        if p.exists():
            return p

    # 2) Start from the user-provided path
    base = Path(json_key_path).expanduser()
    if base.is_absolute():
        candidates = [base]
    else:
        project_root = Path(__file__).resolve().parents[1]
        candidates = [
            base,                              # as given, relative to CWD
            Path.cwd() / base,                 # explicit CWD
            project_root / base,               # project + given path
            project_root / "resources" / base.name,  # project/resources/<file>
        ]

    for c in candidates:
        c = c.resolve()
        if str(c) not in tried:
            tried.append(str(c))
        if c.exists():
            return c

    # Nothing worked
    msg = "Could not find Google service account JSON. Tried:\n  - " + "\n  - ".join(tried)
    raise FileNotFoundError(msg)


#Load data with optional simplification
def load_data_from_gsheet(json_key_path: str, sheet_name: str, simplify: bool = False) -> pd.DataFrame:
    """
    Load a Google Sheet (by spreadsheet title) into a DataFrame.

    Args:
        json_key_path: Path or filename to the service account JSON.
                       You can just pass "cypriot_pos_credentials.json"
                       if it's placed in project_root/resources/.
        sheet_name:    Spreadsheet title to open (first worksheet is used).
        simplify:      If True, maps POS tags via simplify_tag.

    Returns:
        pandas.DataFrame with the sheet data.
    """
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]

    creds_path = _resolve_creds_path(json_key_path)
    creds = ServiceAccountCredentials.from_json_keyfile_name(str(creds_path), scope)
    client = gspread.authorize(creds)

    # Open spreadsheet by title and take the first worksheet
    sheet = client.open(sheet_name).sheet1
    data = sheet.get_all_records()
    df = pd.DataFrame(data)

    if simplify and "POS Tag" in df.columns:
        df["POS Tag"] = df["POS Tag"].apply(simplify_tag)

    return df


#Group tokens into sentences
def group_sentences(df: pd.DataFrame):
    """
    Group rows into sentences as lists of (word, tag) tuples.
    Requires columns: 'Sentence ID', 'Word', 'POS Tag'.
    """
    sentences = []
    for _, group in df.groupby("Sentence ID", sort=True):
        sentence = list(zip(group["Word"], group["POS Tag"]))
        sentences.append(sentence)
    return sentences


#Preview / quick test (optional)
if __name__ == "__main__":
    #With the resolver, you can just pass the filename
    df = load_data_from_gsheet("cypriot_pos_credentials.json", "Data", simplify=False)
    sentences = group_sentences(df)

    print("🔍 Sample sentence:")
    if sentences:
        print(sentences[0])
    else:
        print("(no sentences found)")

    print(f"\nTotal sentences: {len(sentences)}")

    print("\n🧪 Custom sentence example:")
    print(["η", "γιαγιά", "εψούμνισεν", "κουπέπια"])

