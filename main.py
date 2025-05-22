import pandas as pd
import re
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Load German NLP model (not directly used but available)
nlp = spacy.load("de_core_news_md")

# Load input data
df = pd.read_csv("ds_ner_test_case.csv")

# --- STORAGE EXTRACTION ---

def extract_storage(text):
    if not isinstance(text, str):
        return None
    matches = re.findall(r"\b\d+\s?(GB|TB)\b", text, re.IGNORECASE)
    for match in matches:
        context_match = re.search(r"(\d+\s?(GB|TB))", text, re.IGNORECASE)
        if context_match:
            span = context_match.span()
            context = text[max(0, span[0]-15):span[1]+15].lower()
            if not any(kw in context for kw in ["ram", "arbeitsspeicher", "ddr"]):
                return context_match.group(1)
    return None

def extract_storage_from_priority(row):
    for field in ["headline", "highlights", "description"]:
        storage = extract_storage(row.get(field, ""))
        if storage:
            return storage
    return None

df["storage"] = df.apply(extract_storage_from_priority, axis=1)

# --- COLOR EXTRACTION ---

color_list = [
    "schwarz", "weiß", "weiss", "rot", "blau", "gelb", "grün", "gruen", "grau", "braun",
    "lila", "türkis", "tuerkis", "koralle", "silber", "anthrazit", "rauchgrau",
    "aluminium", "hellblau", "dunkelblau", "violett", "neongelb", "wildkirsche",
    "black", "white", "red", "blue", "green", "yellow", "gray", "grey", "silver",
    "gold", "pink", "purple", "orange", "brown", "beige", "turquoise", "navy",
    "ruby", "graphite", "silber-grau", "silber-schwarz", "weiß-rot", "schwarz-neongelb", "silber/weiß"
]
color_pattern = re.compile(r"\b(" + "|".join(map(re.escape, color_list)) + r")\b", re.IGNORECASE)

def extract_color(text):
    if not isinstance(text, str):
        return None
    match = color_pattern.search(text.lower())
    if match:
        return match.group(1)
    return None

def extract_color_from_priority(row):
    for field in ["headline", "highlights", "description"]:
        color = extract_color(row.get(field, ""))
        if color:
            return color
    return None

df["color_extracted"] = df.apply(extract_color_from_priority, axis=1)

# --- BRAND EXTRACTION ---

def extract_brand_from_headline(headline):
    if isinstance(headline, str) and headline.strip():
        return headline.strip().split()[0]
    return None

df["brand_extracted"] = df["headline"].apply(extract_brand_from_headline)

# --- STORAGE CONVERSION TO GB ---

def convert_storage_to_gb(storage_str):
    if pd.isnull(storage_str):
        return None
    matches = re.findall(r'(\d+(?:[.,]\d+)?)\s*(TB|GB)', storage_str.upper())
    total_gb = 0
    for value, unit in matches:
        value = float(value.replace(',', '.'))
        if unit == 'TB':
            value *= 1024
        total_gb += value
    return int(total_gb) if total_gb else None

# --- COLOR STANDARDIZATION ---

color_mapping = {
    "black": "Schwarz", "schwarz": "Schwarz",
    "white": "Weiß", "weiß": "Weiß",
    "blue": "Blau", "blau": "Blau",
    "red": "Rot", "rot": "Rot",
    "green": "Grün", "grün": "Grün",
    "silver": "Silber", "silber": "Silber",
    "gray": "Grau", "grau": "Grau",
    "gold": "Gold", "pink": "Pink",
    "lila": "Lila", "purple": "Lila",
    "orange": "Orange", "brown": "Braun", "braun": "Braun"
}

def standardize_color(value):
    if pd.isna(value):
        return None
    return color_mapping.get(str(value).strip().lower(), value)

df["color_extracted_std"] = df["color_extracted"].apply(standardize_color)
df["farbe_filled"] = df["Farbe"].combine_first(df["color_extracted_std"])
df["storage_filled"] = df["Speicherkapazität"].combine_first(df["storage"])
df["storage_gb"] = df["storage_filled"].apply(convert_storage_to_gb)

# Save cleaned outputs
df.to_csv("ds_ner_test_case_cleaned_combined.csv", index=False)
df[["headline", "brand_extracted", "farbe_filled", "storage_gb"]].to_csv("enriched_products.csv", index=False)

# --- CLASSIFICATION MODELING ---

# Load enriched dataset
df = pd.read_csv('enriched_products.csv')

# Filter for valid rows
df_brand = df.dropna(subset=['brand_extracted', 'headline'])
df_color = df.dropna(subset=['farbe_filled', 'headline'])
df_storage = df.dropna(subset=['storage_gb', 'headline']).copy()

# Bin storage values
bins = [0, 16, 64, 128, 256, 512, 1024, 2048, np.inf]
labels = ['<=16GB', '17-64GB', '65-128GB', '129-256GB', '257-512GB', '513-1024GB', '1025-2048GB', '>2048GB']
df_storage['storage_class'] = pd.cut(df_storage['storage_gb'], bins=bins, labels=labels)

def train_eval_classifier(df, target_col):
    counts = df[target_col].value_counts()
    valid_classes = counts[counts >= 2].index
    df_filtered = df[df[target_col].isin(valid_classes)]

    X_train, X_test, y_train, y_test = train_test_split(
        df_filtered['headline'], df_filtered[target_col],
        test_size=0.2, random_state=42, stratify=df_filtered[target_col]
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)
    print(f"--- Evaluation for {target_col} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return clf, vectorizer

# Train and evaluate classifiers
clf_brand, vec_brand = train_eval_classifier(df_brand, 'brand_extracted')
clf_color, vec_color = train_eval_classifier(df_color, 'farbe_filled')
clf_storage, vec_storage = train_eval_classifier(df_storage, 'storage_class')
