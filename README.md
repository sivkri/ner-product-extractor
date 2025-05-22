# Product Specification Extractor (NER) from German Product Data

This project extracts structured product specifications — **brand**, **storage capacity**, and **color** — from unstructured German-language product listings using a combination of **regex**, **rule-based NLP**, and **machine learning classification**.


## project structure
main.py                  # Main script: extraction + ML classification

Dockerfile               # Docker config

requirements.txt         # Python dependencies

ds_ner_test_case.csv     # Input file

             


It supports:

- Named Entity Recognition (NER) for storage, color, and brand
- Standardization and enrichment of fields
- Text classification (TF-IDF + Logistic Regression) for brand, color, and storage class prediction
- Dockerized environment for reproducibility

---


# Build the Docker image
docker build -t ner-product-extractor .

# Run the container
docker run --rm -v "$(pwd)":/app ner-product-extractor

---


ML Model : Trains a logistic regression model using TF-IDF features on headline for: Brand, Color, Storage Class (binned GB values)

Storage Classes Used:

<=16GB

17-64GB

65-128GB

129-256GB

257-512GB

513-1024GB

1025-2048GB

>2048GB
