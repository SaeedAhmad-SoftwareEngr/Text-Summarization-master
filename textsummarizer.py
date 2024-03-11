# Sreamlit Final App, you can run by Streamlit run textsummarizer.py

import streamlit as st
from summarizer import Summarizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import base64
import fitz
import io
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
import heapq
import re
import requests
from transformers import pipeline

# Model and tokenizer loading
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, offload_folder=r"D:\Text-Summarization-master\offloads", torch_dtype=torch.float32)

# File loader and preprocessing using PyMuPDF
def file_preprocessing(file_content):
    doc = fitz.Document(stream=io.BytesIO(file_content), filetype="pdf")
    final_texts = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text()
        final_texts += text
    return final_texts

# LLM pipeline
def llm_pipeline(input_text, num_lines):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50,
        device=0 if torch.cuda.is_available() else -1
    )
    result = pipe_sum(input_text, max_length=num_lines*50, min_length=num_lines*10)
    result = result[0]['summary_text']
    return result

# NLTK summarizer function
def nltk_summarizer(docx, num_sentences):
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(docx)
    freqTable = dict()

    for word in words:
        word = word.lower()
        if word not in stopWords:
            if word in freqTable:
                freqTable[word] += 1
            else:
                freqTable[word] = 1

    sentence_list = sent_tokenize(docx)
    max_freq = max(freqTable.values())
    for word in freqTable.keys():
        freqTable[word] = (freqTable[word] / max_freq)

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in freqTable.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = freqTable[word]
                    else:
                        sentence_scores[sent] += freqTable[word]

    # Use num_sentences parameter here
    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary

# Function to count words
def count_words(text):
    words = text.split()
    return len(words)

# Function to display PDF of a given file
def display_pdf(file_content):
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64.b64encode(file_content).decode("utf-8")}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Function to count words in PDF
def count_words_pdf(text):
    words = text.split()
    return len(words)

# API Summarizer function
def api_summarizer(input_text, language):
    url = "https://article-extractor-and-summarizer.p.rapidapi.com/summarize-text"
    payload = {
        "lang": language,
        "text": input_text
    }
    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": "e8962e6879msh816cb4de8995507p1e0d1ejsn7bcca960c382",
        "X-RapidAPI-Host": "article-extractor-and-summarizer.p.rapidapi.com"
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json().get('summary', '')

# Streamlit code
st.set_page_config(layout="wide")

def main():
    st.title("Text Summarization App using NLP")

    # Use sidebar for options
    st.sidebar.header("Options")
    option = st.sidebar.selectbox("Select Input Type", ("Text", "Document"))

    summarization_methods = ["LaMini Model", "NLTK", "NLP"]
    selected_method = st.sidebar.selectbox("Select Summarization Method", summarization_methods)

    language_options = ["en", "ur"]
    if selected_method == "NLP":
        language = st.sidebar.selectbox("Select Language", language_options)
    else:
        language = "en"

    if option == "Text":
        # Text Input
        input_text = st.text_area("Enter your text here", height=200)
        st.info(f"Word Count: {count_words(input_text)} words")

        num_lines = st.slider("Select the number of lines for summary", 1, 20, 5)

        if st.button("Summarize"):
            st.info("Input Text")

            if selected_method == "LaMini Model":
                summary = llm_pipeline(input_text, num_lines)
            elif selected_method == "NLTK":
                summary = nltk_summarizer(input_text, num_lines)
            elif selected_method == "NLP":
                summary = api_summarizer(input_text, language)

            st.info("Summarization Complete")
            st.success(summary)
            st.info(f"Word Count after Summarization: {count_words(summary)} words")

    elif option == "Document":
        # PDF File Upload
        uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'], key="pdf_upload")

        num_lines = st.slider("Select the number of lines for summary", 1, 20, 5)

        if uploaded_file is not None:
            if st.button("Summarize"):
                col1, col2 = st.columns(2)

                file_content = uploaded_file.read()
                input_text = file_preprocessing(file_content)

                with col1:
                    st.info("Uploaded File")
                    display_pdf(file_content)

                with col2:
                    st.info(f"Word Count in PDF: {count_words_pdf(input_text)} words")

                    if selected_method == "LaMini Model":
                        summary = llm_pipeline(input_text, num_lines)
                    elif selected_method == "NLTK":
                        summary = nltk_summarizer(input_text, num_lines)
                    elif selected_method == "NLP":
                        summary = api_summarizer(input_text, language)

                    st.info("Summarization Complete")
                    st.success(summary)
                    st.info(f"Word Count after Summarization: {count_words(summary)} words")

if __name__ == "__main__":
    main()


