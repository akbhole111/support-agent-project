import kagglehub
import pandas as pd
import os
import json
from datasets import load_dataset

import math

def clean_nans(records):
    """Recursively replace float NaN with None in a list of dicts."""
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, float) and math.isnan(v):
                rec[k] = None
    return records

# ============================================================
# 1. DOWNLOAD DATASETS (your existing script)
# ============================================================
olist_path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
tickets_path = kagglehub.dataset_download("suraj520/customer-support-ticket-dataset")

print("Olist path:", olist_path)
print("Tickets path:", tickets_path)

OUTPUT_DIR = "D:/support-agent-project/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. CLEAN ORDERS DATA (for the order-status tool)
# ============================================================
orders = pd.read_csv(os.path.join(olist_path, "olist_orders_dataset.csv"))
items = pd.read_csv(os.path.join(olist_path, "olist_order_items_dataset.csv"))
customers = pd.read_csv(os.path.join(olist_path, "olist_customers_dataset.csv"))
products = pd.read_csv(os.path.join(olist_path, "olist_products_dataset.csv"))

# Get one representative item per order (product + price)
order_items_slim = (
    items.groupby("order_id")
    .agg(product_id=("product_id", "first"),
         price=("price", "sum"),
         freight_value=("freight_value", "sum"))
    .reset_index()
)

# Merge everything into one flat table
merged = (
    orders
    .merge(order_items_slim, on="order_id", how="left")
    .merge(customers[["customer_id", "customer_city", "customer_state"]], on="customer_id", how="left")
    .merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
)

# Keep only relevant columns, rename for clarity
merged = merged[[
    "order_id", "customer_id", "order_status",
    "product_category_name", "price", "freight_value",
    "order_purchase_timestamp", "order_delivered_customer_date",
    "order_estimated_delivery_date", "customer_city", "customer_state"
]].rename(columns={
    "order_purchase_timestamp": "purchase_date",
    "order_delivered_customer_date": "delivered_date",
    "order_estimated_delivery_date": "estimated_delivery_date",
    "product_category_name": "item_category"
})

# Drop rows with no order_status or product info
merged = merged.dropna(subset=["order_status"])

# Sample a manageable subset for the mock API (e.g. 500 orders across varied statuses)
# NOTE: using a plain loop + concat instead of groupby().apply() —
# groupby().apply() on newer pandas versions silently drops the
# grouping column from the result, which breaks downstream code
# that reads sampled["order_status"].
sampled_groups = []
for status, grp in merged.groupby("order_status"):
    sampled_groups.append(grp.sample(min(len(grp), 100), random_state=42))
sampled = pd.concat(sampled_groups).reset_index(drop=True)

orders_records = sampled.to_dict(orient="records")
orders_records = clean_nans(orders_records)

with open(os.path.join(OUTPUT_DIR, "orders.json"), "w") as f:
    json.dump(orders_records, f, indent=2, default=str)

print(f"Saved {len(orders_records)} orders to data/orders.json")
print("Status breakdown:\n", sampled["order_status"].value_counts())

# ============================================================
# 3. LOAD BITEXT CUSTOMER SUPPORT DATASET (real Q&A pairs)
# ============================================================
bitext = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
df = bitext["train"].to_pandas()

# Columns: flags, instruction, category, intent, response
faq_docs = []
for _, row in df.iterrows():
    faq_docs.append({
        "category": row["category"],
        "intent": row["intent"],
        "question": row["instruction"],
        "answer": row["response"]
    })

faq_df = pd.DataFrame(faq_docs).drop_duplicates(subset=["question", "answer"])

# Sample a manageable KB size — e.g. 10 per intent across all 27 intents (~270 docs)
# Same fix as above: plain loop + concat instead of groupby().apply(),
# so the "intent" column survives into faq_sample_df.
sampled_groups = []
for intent, grp in faq_df.groupby("intent"):
    sampled_groups.append(grp.sample(min(len(grp), 10), random_state=42))
faq_sample_df = pd.concat(sampled_groups).reset_index(drop=True)

faq_sample = faq_sample_df.to_dict(orient="records")
faq_sample = clean_nans(faq_sample)

with open(os.path.join(OUTPUT_DIR, "faqs.json"), "w") as f:
    json.dump(faq_sample, f, indent=2, default=str)

print(f"Saved {len(faq_sample)} FAQ documents to data/faqs.json")
print("Category breakdown:\n", faq_sample_df["category"].value_counts())
print("Intent breakdown:\n", faq_sample_df["intent"].value_counts())