"""Streamlit interface for the zero-shot news classifier."""
import json

import pandas as pd
import streamlit as st

from classifier import NewsClassifier, validate_input

st.set_page_config(page_title="Zero-Shot News Classifier", page_icon="📰", layout="centered")


@st.cache_resource
def get_classifier():
    return NewsClassifier()


st.title("Zero-Shot News Classifier")
st.write("Classify a passage using your own topic labels.")
with st.form("classification"):
    text = st.text_area("Passage", "The team won the final cricket match.", height=130)
    raw_labels = st.text_input("Candidate topics, separated by commas", "sports, technology, politics, education")
    multi_label = st.checkbox("Allow multiple topics", value=False)
    submitted = st.form_submit_button("Classify passage", type="primary")
if submitted:
    try:
        labels = raw_labels.split(",")
        validate_input(text, labels)
        with st.spinner("Analyzing the passage. The first run downloads the model."):
            result = get_classifier().classify(text, labels, multi_label=multi_label)
        st.session_state["result"] = result
    except (ValueError, OSError, RuntimeError) as error:
        st.session_state.pop("result", None)
        st.error(str(error))
if "result" in st.session_state:
    result = st.session_state["result"]
    if not result["multi_label"]:
        st.subheader("Suggested topic: " + result["labels"][0])
    else:
        st.subheader("Topic relevance")
    table = pd.DataFrame({"Topic": result["labels"], "Score": result["scores"]})
    st.dataframe(table, hide_index=True, column_config={"Score": st.column_config.NumberColumn(format="%.4f")}, width="stretch")
    st.bar_chart(table.set_index("Topic"), horizontal=True)
    st.caption("Scores express model preference. They are not a guarantee of correctness.")
    st.download_button("Download results as JSON", json.dumps(result, indent=2), "classification.json", "application/json")
    with st.expander("Model and scoring"):
        st.write("BART-MNLI compares the passage with one hypothesis per topic. Single-topic scores sum to one across the supplied labels; multiple-topic scores are normalized independently.")
        st.code(result["model"])
