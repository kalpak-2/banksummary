from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import openai
import io
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Allow Lovable frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with specific domain when live
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def classify_transaction(description):
    description = str(description).lower()
    if "upi" in description:
        return "UPI"
    elif "neft" in description:
        return "NEFT"
    elif "rtgs" in description:
        return "RTGS"
    elif "imps" in description:
        return "IMPS"
    elif "cash" in description or "atm" in description or "deposit" in description:
        return "Cash/Deposit"
    elif "pos" in description or "card" in description:
        return "Card/POS"
    else:
        return "Other"

def classify_amount(amount):
    if amount < 1000:
        return "Small (<₹1k)"
    elif amount < 10000:
        return "Medium (₹1k–₹10k)"
    elif amount < 50000:
        return "Large (₹10k–₹50k)"
    else:
        return "Significant (₹50k+)"

@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_excel(io.BytesIO(content))

    # Clean and prepare
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df['ExecutionType'] = df['Description'].apply(classify_transaction)
    df['AmountBucket'] = df['Amount'].abs().apply(classify_amount)

    # Grouped summaries
    exec_type_counts = df['ExecutionType'].value_counts().to_dict()
    amount_buckets = df['AmountBucket'].value_counts().to_dict()
    total_credits = df[df['Amount'] > 0]['Amount'].sum()
    total_debits = df[df['Amount'] < 0]['Amount'].sum()

    # Ask GPT for summary
    prompt = f"""
    The client has uploaded a bank statement. Here are the stats:
    - Total Credits: ₹{total_credits:.2f}
    - Total Debits: ₹{total_debits:.2f}
    - Execution Types: {exec_type_counts}
    - Amount Categories: {amount_buckets}

    Generate a brief summary for a Chartered Accountant preparing ITR.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return {
        "summary": response["choices"][0]["message"]["content"],
        "execution_types": exec_type_counts,
        "amount_buckets": amount_buckets,
        "total_credits": total_credits,
        "total_debits": total_debits,
    }
