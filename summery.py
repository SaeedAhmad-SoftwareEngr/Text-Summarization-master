# ---FINAL CODE ---
import io
import base64
import heapq
from flask import Flask, render_template, request, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import fitz
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import torch
import requests
import PyPDF2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'arslanasdkjksdfhhhh'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(
    checkpoint, offload_folder=r"D:\Text-Summarization-master\offloads", torch_dtype=torch.float32
)


def file_preprocessing(file_content):
    doc = fitz.Document(stream=io.BytesIO(file_content), filetype="pdf")
    final_texts = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text()
        final_texts += text
    return final_texts


def llm_pipeline(input_text, num_lines):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50,
        device=0 if torch.cuda.is_available() else -1
    )
    result = pipe_sum(input_text, max_length=num_lines * 50, min_length=num_lines * 10)
    result = result[0]['summary_text']
    return result


def nltk_summarizer(docx, num_sentences):
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(docx)
    freqTable = dict()

    for word in words:
        word = word.lower()
        if word not in stopWords:
            freqTable[word] = freqTable.get(word, 0) + 1

    sentence_list = sent_tokenize(docx)
    max_freq = max(freqTable.values())
    for word in freqTable:
        freqTable[word] = freqTable[word] / max_freq

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in freqTable:
                if len(sent.split(' ')) < 30:
                    sentence_scores[sent] = sentence_scores.get(sent, 0) + freqTable[word]

    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary


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


def count_words(text):
    words = text.split()
    return len(words)


def count_words_pdf(text):
    words = text.split()
    return len(words)


def summarize_text(input_text, num_sentences, summarization_method, language=None, pdf_content=None):
    original_word_count = count_words(input_text)

    if summarization_method == "LaMini Model":
        summary = llm_pipeline(input_text, num_sentences)
    elif summarization_method == "NLTK":
        summary = nltk_summarizer(input_text, num_sentences)
    elif summarization_method == "NLP":
        summary = api_summarizer(input_text, language)
    elif summarization_method == "PDF":
        input_text = file_preprocessing(pdf_content)
        summary = llm_pipeline(input_text, num_sentences)
    else:
        return "Method not supported", 0, 0

    summarized_word_count = count_words(summary)
    return summary, original_word_count, summarized_word_count


@app.route('/', methods=['GET', 'POST'])
def summarizer():
    if request.method == 'POST':
        input_type = request.form['input_type']
        summarization_method = request.form['summarization_method']

        # Initialize input_text before the conditional block
        input_text = ""

        if input_type == "Text":
            input_text = request.form['input_text']
            num_sentences = int(request.form['num_sentences'])
            language = request.form.get('language', '')
            summary, word_count_before, word_count_after = summarize_text(input_text, num_sentences, summarization_method, language)
        elif input_type == "Document":
            pdf_file = request.files['pdf_file']
            if pdf_file:
                pdf_content = pdf_file.read()
                num_sentences = int(request.form['num_sentences'])
                language = request.form.get('language', '')
                input_text = file_preprocessing(pdf_content)  # Set input_text for document (PDF)
                summary, word_count_before, word_count_after = summarize_text(input_text, num_sentences, summarization_method, language, pdf_content)
            else:
                return 'No PDF file provided'
        else:
            return 'Invalid input type'

        return render_template(
            'summarizer.html', input_text=input_text, output_summary=summary,
            word_count_before=word_count_before, word_count_after=word_count_after
        )

    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('summarizer_page'))

        return 'Invalid username or password'

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return 'Username already exists, please choose another username.'

        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))

    return render_template('signup.html', title="Sign-up")


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/summarizer', methods=['GET', 'POST'])
@login_required
def summarizer_page():
    input_text = ""  # Initialize input_text with an empty string

    if request.method == 'POST':
        input_type = request.form['input_type']
        summarization_method = request.form['summarization_method']

        if input_type == "Text":
            input_text = request.form['input_text']
            num_sentences = int(request.form['num_sentences'])
            language = request.form.get('language', '')
            summary, word_count_before, word_count_after = summarize_text(input_text, num_sentences, summarization_method, language)
        elif input_type == "Document":
            pdf_file = request.files['pdf_file']
            if pdf_file:
                pdf_content = pdf_file.read()
                num_sentences = int(request.form['num_sentences'])
                language = request.form.get('language', '')
                input_text = file_preprocessing(pdf_content)  # Set input_text for document (PDF)
                summary, word_count_before, word_count_after = summarize_text(input_text, num_sentences, summarization_method, language, pdf_content)
            else:
                return 'No PDF file provided'
        else:
            return 'Invalid input type'

        return render_template(
            'summarizer.html', input_text=input_text, output_summary=summary,
            word_count_before=word_count_before, word_count_after=word_count_after
        )

    return render_template('summarizer.html')


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host='127.0.0.1')



