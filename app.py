# ============================================================
# PROJECT 1 — SENTIMENT ANALYSER
# Step 5: Streamlit UI
# ============================================================
# Install: pip install streamlit transformers torch scikit-learn
# Run:     streamlit run app.py
# ============================================================

import streamlit as st
import torch
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="Sentiment Analyser",
    page_icon="💬",
    layout="centered"
)

# ------------------------------------------------------------
# LOAD MODELS (cached so they only load once)
# ------------------------------------------------------------

@st.cache_resource
def load_bert_model():
    MODEL_NAME = "aish2067/distilbert-sentiment-reviews"  # ← your HF repo
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_baseline():
    with open("baseline_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data

# ------------------------------------------------------------
# PREDICTION FUNCTIONS
# ------------------------------------------------------------

LABELS     = ['Negative','Positive']
LABEL_MAP  = {1: 'Negative', 2: 'Positive'}
EMOJI_MAP  = {'Negative': '😠',  'Positive': '😊'}
COLOR_MAP  = {'Negative': '#E24B4A','Positive': '#639922'}

def predict_bert(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1)[0]

    pred_idx   = int(torch.argmax(probs).item())
    labels     = ['Negative',  'Positive']
    pred_label = labels[pred_idx]
    confidence = float(probs[pred_idx].item())
    all_probs  = {labels[i]: float(probs[i].item()) for i in range(2)}

    return pred_label, confidence, all_probs

# ------------------------------------------------------------
# UI — HEADER
# ------------------------------------------------------------

st.title("💬 Customer Review Sentiment Analyser")
st.markdown(
    "Paste any product review below and get an instant sentiment prediction "
    "using a fine-tuned **DistilBERT** model."
)
st.divider()

# ------------------------------------------------------------
# LOAD MODELS WITH SPINNER
# ------------------------------------------------------------

with st.spinner("Loading models..."):
    try:
        tokenizer, bert_model = load_bert_model()
        model_loaded = True
    except Exception as e:
        st.error(f"Could not load DistilBERT model: {e}\n\nMake sure the `distilbert_sentiment/` folder is in the same directory as app.py")
        model_loaded = False

# ------------------------------------------------------------
# TAB 1: SINGLE REVIEW
# ------------------------------------------------------------

tab1, tab2 = st.tabs(["Single Review", "Batch Analysis"])

with tab1:
    st.subheader("Analyse a single review")

    review_input = st.text_area(
        "Paste your review here:",
        placeholder="e.g. This product is absolutely amazing! Best purchase I've made all year.",
        height=150
    )

    if st.button("Analyse Sentiment", type="primary", disabled=not model_loaded):
        if not review_input.strip():
            st.warning("Please enter a review first.")
        else:
            with st.spinner("Analysing..."):
                label, confidence, all_probs = predict_bert(
                    review_input, tokenizer, bert_model)

            # Result card
            emoji = EMOJI_MAP[label]
            color = COLOR_MAP[label]

            st.markdown(f"""
            <div style="
                background: {color}18;
                border-left: 4px solid {color};
                border-radius: 8px;
                padding: 16px 20px;
                margin: 16px 0;
            ">
                <h2 style="margin:0; color:{color}">{emoji} {label}</h2>
                <p style="margin:4px 0 0; color:#666">Confidence: {confidence*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

            # Confidence bars for all 3 classes
            st.markdown("**Confidence breakdown:**")
            for lbl, prob in all_probs.items():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.write(f"{EMOJI_MAP[lbl]} {lbl}")
                with col2:
                    st.progress(prob, text=f"{prob*100:.1f}%")


# ------------------------------------------------------------
# TAB 2: BATCH ANALYSIS
# ------------------------------------------------------------

with tab2:
    st.subheader("Analyse multiple reviews at once")
    st.markdown("Enter one review per line:")

    batch_input = st.text_area(
        "Reviews (one per line):",
        placeholder="This product is great!\nTerrible quality, broke after a week.\nIt's okay, nothing special.",
        height=200
    )

    if st.button("Analyse All", type="primary", disabled=not model_loaded):
        reviews = [r.strip() for r in batch_input.strip().split('\n') if r.strip()]

        if not reviews:
            st.warning("Please enter at least one review.")
        else:
            with st.spinner(f"Analysing {len(reviews)} reviews..."):
                results = []
                for review in reviews:
                    label, confidence, _ = predict_bert(review, tokenizer, bert_model)
                    results.append({
                        'Review': review[:80] + '...' if len(review) > 80 else review,
                        'Sentiment': f"{EMOJI_MAP[label]} {label}",
                        'Confidence': f"{confidence*100:.1f}%"
                    })

            # Results table
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True, hide_index=True)

            # Sentiment breakdown chart
            labels_only = [r['Sentiment'].split()[-1] for r in results]
            counts = pd.Series(labels_only).value_counts()

            fig, ax = plt.subplots(figsize=(5, 3))
            colors  = [COLOR_MAP.get(l, '#888780') for l in counts.index]
            ax.bar(counts.index, counts.values, color=colors, width=0.4, edgecolor='none')
            ax.set_title('Sentiment Breakdown', fontsize=12)
            ax.set_ylabel('Count')
            ax.spines[['top', 'right']].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)

            # Summary stats
            total    = len(results)
            pos      = labels_only.count('Positive')
            neg      = labels_only.count('Negative')

            col1, col2, col3 = st.columns(3)
            col1.metric("Positive", f"{pos}", f"{pos/total*100:.0f}%")
            col3.metric(" Negative", f"{neg}", f"{neg/total*100:.0f}%")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.divider()
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:13px'>"
    "Built with DistilBERT + Streamlit &nbsp;|&nbsp; "
    "Trained on Amazon Product Reviews &nbsp;|&nbsp; "
    "3-class classification: Negative / Neutral / Positive"
    "</p>",
    unsafe_allow_html=True
)