"""Import the libraries."""

import base64
import io
import pickle
import random
import re
import string
import time
import timeit
import uuid

import gdown
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
import torch.nn.functional as F
import tweepy as tw
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from stqdm import stqdm
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from apps import training_model
from apps.indonlu.utils.data_utils import (
    HateSpeechClassificationDataLoader,
    MultiLabelHateSpeechClassificationDataset,
)
from apps.indonlu.utils.forward_fn import forward_sequence_classification
from apps.indonlu.utils.metrics import multi_label_hate_speech_classification_metrics_fn

# Load trained tokenizer with pickle
tokenizer_path = "./models/tokenizer.pkl"
with open(tokenizer_path, "rb") as handle:
    tokenizer_trained = pickle.load(handle)

# Instantiate trained model
config = BertConfig.from_pretrained("./models/config.json")

# url = "https://drive.google.com/"
output = "./models/pytorch_model.bin"
# gdown.download(url, output, quiet=False)
model = BertForSequenceClassification.from_pretrained(output, config=config)

preprocess_input = lambda x: preprocessing(x)

# -----------------
# Define Components
# -----------------


def get_pd_image_download_link(filename, text):
    """Generates a link to download pandas image from a given filename."""

    buffered = io.BytesIO()

    plt.savefig(buffered)
    st.pyplot()
    buffered.seek(0)

    # Styling custom button with css
    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub("\d+", "", button_uuid)

    custom_css = f"""
        <style>
            #{button_id} {{
                font-weight: 400;
                line-height: 1.6;

                position: relative;

                display: inline-flex;

                width: auto;
                margin: 0;
                padding: .25rem .75rem;
                margin-bottom: 1rem;

                text-decoration: none;

                color: inherit;
                background-color: rgb(11, 27, 50);
                border: 1px solid rgba(238, 238, 238, .2);
                border-radius: .25rem;

                -webkit-box-align: center;
                -webkit-box-pack: center;
                align-items: center;
                justify-content: center;
            }}
            #{button_id}:hover {{
                border-color: #B0CDF1;
            }}
            #{button_id}:active {{
                box-shadow: #5C718C 0px 1px 4px, #5C718C 0px 0px 0px 3px;
                background-color: #B0CDF1;
                color: inherit !important;
            }}
            #{button_id}:focus {{
                border-color: #B0CDF1;
                box-shadow: #5C718C 0px 1px 4px, #5C718C 0px 0px 0px 3px;
                color: #B0CDF1;
            }}
        </style>"""

    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = (
        custom_css
        + f'<a href="data:file/txt;base64,{img_str}" download="{filename}" id="{button_id}">{text}</a>'
    )
    return href


def get_plt_image_download_link(fig, filename, text):
    """Generates a link to download plot image from a given filename."""

    buffered = io.BytesIO()

    fig.savefig(buffered)
    buffered.seek(0)

    # Styling custom button with css
    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub("\d+", "", button_uuid)

    custom_css = f"""
        <style>
            #{button_id} {{
                font-weight: 400;
                line-height: 1.6;

                position: relative;

                display: inline-flex;

                width: auto;
                margin: 0;
                padding: .25rem .75rem;
                margin-bottom: 1rem;

                text-decoration: none;

                color: inherit;
                background-color: rgb(11, 27, 50);
                border: 1px solid rgba(238, 238, 238, .2);
                border-radius: .25rem;

                -webkit-box-align: center;
                -webkit-box-pack: center;
                align-items: center;
                justify-content: center;
            }}
            #{button_id}:hover {{
                border-color: #B0CDF1;
            }}
            #{button_id}:active {{
                box-shadow: #5C718C 0px 1px 4px, #5C718C 0px 0px 0px 3px;
                background-color: #B0CDF1;
                color: inherit !important;
            }}
            #{button_id}:focus {{
                border-color: #B0CDF1;
                box-shadow: #5C718C 0px 1px 4px, #5C718C 0px 0px 0px 3px;
                color: #B0CDF1;
            }}
        </style>"""

    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = (
        custom_css
        + f'<a href="data:file/txt;base64,{img_str}" download="{filename}" id="{button_id}">{text}</a>'
    )
    return href


def get_table_download_df(df, filename, text):
    """Generates a link allowing the data in a given pandas dataframe to be downloaded."""

    # Styling custom button with css
    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub("\d+", "", button_uuid)

    custom_css = f"""
        <style>
            #{button_id} {{
                font-weight: 400;
                line-height: 1.6;

                position: relative;

                display: inline-flex;

                width: auto;
                margin: 0;
                padding: .25rem .75rem;
                margin-bottom: 1rem;

                text-decoration: none;

                color: inherit;
                background-color: rgb(11, 27, 50);
                border: 1px solid rgba(238, 238, 238, .2);
                border-radius: .25rem;

                -webkit-box-align: center;
                -webkit-box-pack: center;
                align-items: center;
                justify-content: center;
            }}
            #{button_id}:hover {{
                border-color: #B0CDF1;
            }}
            #{button_id}:active {{
                box-shadow: #5C718C 0px 1px 4px, #5C718C 0px 0px 0px 3px;
                background-color: #B0CDF1;
                color: inherit !important;
            }}
            #{button_id}:focus {{
                border-color: #B0CDF1;
                box-shadow: #5C718C 0px 1px 4px, #5C718C 0px 0px 0px 3px;
                color: #B0CDF1;
            }}
        </style>"""

    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # Some strings <-> bytes conversions necessary here
    href = (
        custom_css
        + f'<a href="data:file/csv;base64,{b64}" download="{filename}" id="{button_id}">{text}</a>'
    )
    return href


def get_table_download_link(df, filename, text):
    """Generates a link allowing the data in a given pandas dataframe to be downloaded."""

    # Styling custom button with css
    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub("\d+", "", button_uuid)

    custom_css = f"""
        <style>
            #{button_id} {{
                font-weight: 400;
                line-height: 1.6;

                position: relative;

                display: inline-flex;

                width: auto;
                margin: 0;
                padding: .25rem .75rem;
                margin-bottom: 1rem;

                text-decoration: none;

                color: inherit;
                background-color: rgb(11, 27, 50);
                border: 1px solid rgba(238, 238, 238, .2);
                border-radius: .25rem;

                -webkit-box-align: center;
                -webkit-box-pack: center;
                align-items: center;
                justify-content: center;
            }}
            #{button_id}:hover {{
                border-color: #B0CDF1;
            }}
            #{button_id}:active {{
                box-shadow: #5C718C 0px 1px 4px, #5C718C 0px 0px 0px 3px;
                background-color: #B0CDF1;
                color: inherit !important;
            }}
            #{button_id}:focus {{
                border-color: #B0CDF1;
                box-shadow: #5C718C 0px 1px 4px, #5C718C 0px 0px 0px 3px;
                color: #B0CDF1;
            }}
        </style>"""

    csv = df.to_csv()
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # Some strings <-> bytes conversions necessary here
    href = (
        custom_css
        + f'<a href="data:file/csv;base64,{b64}" download="{filename}" id="{button_id}">{text}</a>'
    )
    return href


# ----------------
# Define Functions
# ----------------


def css(filename):
    """Load css file for styling streamlit website."""

    with open(filename) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def twitter_api(type_of_search, text_input, num_of_tweets):
    """Get twitter data using API Search."""

    # Define access keys and tokens
    consumer_key = st.secrets["consumer_key"]
    consumer_secret = st.secrets["consumer_secret"]
    access_token = st.secrets["access_token"]
    access_token_secret = st.secrets["access_token_secret"]

    # Set up Twitter API access
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)

    # Select keyword or hashtag
    if type_of_search == "Kata Kunci":
        user_word = text_input
    else:
        user_word = "#" + text_input

    user_word = user_word + " -filter:retweets"
    tweets = tw.Cursor(
        api.search_tweets, q=user_word, tweet_mode="extended", lang="id"
    ).items(num_of_tweets)

    # Store data as dataframe
    tweet_metadata = [
        [tweet.id, tweet.created_at, tweet.user.screen_name, tweet.full_text]
        for tweet in tweets
    ]
    df_tweets = pd.DataFrame(
        data=tweet_metadata, columns=["id", "created_at", "user", "full_text"]
    )

    df_tweets["created_dttime"] = df_tweets["created_at"].apply(
        lambda x: x.strftime("%a %b %d %Y %X")
    )

    # Create a new text variable to do manipulations on
    df_tweets["clean_text"] = df_tweets.full_text

    df_new = df_tweets[["created_at", "user", "full_text"]]
    df_new = df_new.rename(columns={"user": "Username", "full_text": "Tweet"})

    return df_tweets, df_new


def single_tweet_labelling(result):
    if result == 1:
        return "Hate Speech Lemah"
    elif result == 2:
        return "Hate Speech Menengah"
    elif result == 3:
        return "Hate Speech Kuat"
    else:
        return "Non Hate Speech"


def multi_tweet_labelling(result):
    if result == 1:
        return "Weak Hate Speech"
    elif result == 2:
        return "Moderate Hate Speech"
    elif result == 3:
        return "Strong Hate Speech"
    else:
        return "Non Hate Speech"


def single_tweet_detection(text):
    encoded_inputs = tokenizer_trained.encode_plus(text, add_special_tokens=True)
    subwords = encoded_inputs["input_ids"]
    subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

    logits = model(subwords)[0]
    label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()
    probability = f"{F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}"
    result = single_tweet_labelling(label)

    return result, probability


def multi_hatespeech_detection(df, data_column):
    label_hs = []
    prob = []

    for index, row in df.iterrows():

        encoded_inputs = tokenizer_trained.encode_plus(
            str(row[data_column]), add_special_tokens=True
        )
        subwords = encoded_inputs["input_ids"]
        subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)
        logits = model(subwords)[0]
        label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()
        probability = f"{F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}%"

        label_hs.append(multi_tweet_labelling(label))
        prob.append(probability)

    df[f"Label"] = label_hs
    df[f"Probability"] = prob
    return df


def set_seed(seed):
    """Set the seed value all over the place to make this reproducible."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def count_param(module, trainable=False):
    """Count number of parameters."""

    if trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())


def get_lr(optimizer):
    """Get the learning rate."""

    for param_group in optimizer.param_groups:
        return param_group["lr"]


def metrics_to_string(metric_dict):
    """Change from metrics evaluation to string."""

    string_list = []

    for key, value in metric_dict.items():
        string_list.append("{}:{:.2f}".format(key, value))
    return " ".join(string_list)


def set_hyperparameter():
    """Set global variable, parameter, and hyperparameter used (Hyperparameter Tuning)."""

    with st.container():
        global batch_size, lr, epochs, attention_probs_dropout_prob, hidden_dropout_prob, max_seq_len, num_labels, num_workers, eps, weight_decay, random_state

        st.header("Konfigurasi Hyperparameter")
        st.info("Atur Hyperparameter dengan mengubah nilai-nilai di bawah.")

        # Iterable
        batch_size = st.selectbox(
            "Batch Size",
            (16, 32),
            help=(
                "Nilai yang menentukan jumlah sampel data yang disebar pada setiap iterasi pelatihan, "
                "semakin besar nilainya maka memori yang digunakan semakin banyak."
            ),
            key="batch_size",
        )
        lr = st.selectbox(
            "Learning Rate",
            (2e-5, 3e-5, 5e-5),
            help=(
                "Nilai yang menentukan seberapa banyak koreksi bobot yang akan diubah, "
                "semakin besar nilainya maka pelatihan akan kurang optimal karena proses yang semakin cepat."
            ),
            key="lr",
        )
        epochs = st.selectbox(
            "Epochs",
            (2, 3, 4),
            help=(
                "Nilai yang menentukan berapa kali pelatihan mengolah seluruh dataset, "
                "semakin besar nilainya maka semakin baik tingkat akurasinya."
            ),
            key="epochs",
        )

        random_state = 0
        eps = 1e-8
        max_seq_len = 512
        num_workers = 1
        attention_probs_dropout_prob = 0.1
        hidden_dropout_prob = 0.1
        num_labels = 4
        weight_decay = 1e-2

        with st.sidebar:
            st.header("Detail Hyperparameter")
            st.caption(
                "Nilai-nilai Hyperparameter yang diatur akan ditampilkan di bawah."
            )
            st.info(
                "Batch Size: "
                + str(batch_size)
                + "\n\nLearning Rate: "
                + str(lr)
                + "\n\nEpoch: "
                + str(epochs)
            )

            st.header("Info Label")
            st.caption("Keterangan label pada dataset akan ditampilkan di bawah.")
            st.info(
                "Non Hate Speech: 0"
                "\n\nHate Speech Lemah: 1"
                "\n\nHate Speech Menengah: 2"
                "\n\nHate Speech Kuat: 3"
            )

    return (
        batch_size,
        lr,
        epochs,
        max_seq_len,
        num_workers,
        eps,
        random_state,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
        num_labels,
        weight_decay,
    )


def show_histogram(df):
    """Show data on Histogram."""

    pd.value_counts(df["HS"]).plot.bar()

    plt.title("Label Comparison")
    plt.xlabel("Label")
    plt.ylabel("Count")

    st.markdown(
        get_pd_image_download_link("histogram.png", "Unduh Diagram"),
        unsafe_allow_html=True,
    )


def show_pie(inner_label, outer_label, inner_df, outer_df, legend_title):
    """Show data on Donut Chart."""

    fig, ax = plt.subplots(figsize=(20, 10), subplot_kw=dict(aspect="equal"))
    width = 0.3
    labels = [x.split()[-1] for x in inner_label]

    def func(pct, allvals):
        absolute = int(round(pct / 100.0 * np.sum(allvals)))
        return "{:.1f}% ({:d})".format(pct, absolute)

    # Inner pie
    inner_pie = ax.pie(
        inner_df,
        autopct=lambda pct: func(pct, inner_df),
        textprops=dict(color="white", weight="bold"),
        pctdistance=0.55,
        wedgeprops={"width": 0.6, "edgecolor": "white"},
    )
    plt.setp(inner_pie)

    # Outer pie
    outer_pie = ax.pie(
        outer_df,
        autopct=lambda pct: func(pct, outer_df),
        textprops=dict(color="white", weight="bold"),
        pctdistance=0.85,
        wedgeprops={"width": 0.3, "edgecolor": "white"},
    )
    plt.setp(outer_pie)

    # Set legend label
    ax.legend(
        labels, title=legend_title, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1)
    )
    ax.set_title("Label Comparison")

    plt.tight_layout()

    st.pyplot(fig)
    st.markdown(
        get_plt_image_download_link(fig, "donut_chart.png", "Unduh Diagram"),
        unsafe_allow_html=True,
    )


def preprocessing(text):
    """Text Preprocessing for single tweet detection."""

    # -------------
    # Data Cleaning
    # -------------
    def data_cleaning(text):
        """Data Cleaning."""

        regrex_pattern = re.compile(
            pattern="["
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Symbols & pictographs
            "\U0001F680-\U0001F6FF"  # Transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
            "]+",
            flags=re.UNICODE,
        )
        emoticon_byte_regex = r"\s*(?:\\x[A-Fa-f0-9]{2})+"
        url_regex = "((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+)||(http\S+))"

        text = regrex_pattern.sub(r"", text)  # Remove emojis
        text = re.sub(emoticon_byte_regex, "", text)  # Remove emoticon bytes
        text = re.sub(url_regex, "", text)  # Remove every url
        text = re.sub(r"(\s)@\w+", "", text)  # Remove whole word if starts with @
        text = re.sub(r"(\s).@\w+", "", text)  # Remove whole word if starts with @
        text = re.sub(
            r"(\s)\w*\d\w*\w+", "", text
        )  # Remove whole word if starts with number
        text = re.sub(r"<[^>]*>", "", text)  # Remove html tags
        text = re.sub(r"@[A-Za-z0-9]+", "", text)  # Remove twitter usernames
        text = re.sub(r"\\n", " ", text)  # Remove every new line '\n'
        text = re.sub(r"https\:\/\/t\.co\/*\w*", "", text)  # Remove https links
        text = re.sub("[^0-9a-zA-Z]", " ", text)  # Remove punctuation
        text = re.sub("\[.*?\]", "", text)  # Removes text in square brackets
        text = re.sub("@[\w\-]+", "", text)  # Remove mentions
        text = re.sub("RT", "", text)  # Remove every retweet symbol
        text = re.sub("USER", "", text)  # Remove every user
        text = re.sub(" URL", " ", text)  # Remove word URL
        text = re.sub(" url", " ", text)  # Remove word url
        text = re.sub("\\+", " ", text)  # Remove backslash
        text = re.sub("\s+", " ", text)  # Remove special regular expression character
        text = re.sub("[^a-zA-Z]", " ", text)  # Remove numbers
        text = re.sub(" +", " ", text)  # Remove extra spaces
        text = re.sub(
            "[%s]" % re.escape(string.punctuation), "", text
        )  # Removes punctuation

        text = (
            text.strip()
        )  # Removes any spaces or specified characters at the start and end of a string

        return text

    text = data_cleaning(text)

    # ------------
    # Case Folding
    # ------------
    def case_folding(text):
        """Lowercase letters."""

        text = text.lower()
        return text

    text = case_folding(text)

    # -------------
    # Normalization
    # -------------
    # Import Slang Dictionary from (Ibrohim & Budi, 2019)
    alay_dict = "https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/new_kamusalay.csv"
    alay_dict = pd.read_csv(alay_dict, encoding="latin-1", header=None)
    alay_dict = alay_dict.rename(columns={0: "original", 1: "replacement"})
    alay_dict_map = dict(zip(alay_dict["original"], alay_dict["replacement"]))

    def normalization(text):
        """Normalization."""

        return " ".join(
            [
                alay_dict_map[word] if word in alay_dict_map else word
                for word in text.split(" ")
            ]
        )

    text = normalization(text)

    return text


def text_preprocessing(df):
    """Text Preprocessing for training the model."""

    if (
        training_model.check_data_cleaning
        or training_model.check_case_folding
        or training_model.check_normalization
        or training_model.check_filtering
        or training_model.check_stemming
    ):
        with st.expander("Text Preprocessing", expanded=True):
            # Import raw dataset as a comparison
            df_raw = "https://raw.githubusercontent.com/ryzanugrah/id-multi-label-hate-speech-and-abusive-language-detection/master/re_multi_label_dataset.csv"
            df_raw = pd.read_csv(df_raw, encoding="utf-8")
            df_raw.rename({"Tweet": "raw_tweet"}, axis=1, inplace=True)

            if training_model.check_data_cleaning:
                # -------------
                # Data Cleaning
                # -------------
                st.subheader("Data Cleaning")
                st.caption(
                    "Menghapus beberapa karakter yang tidak dibutuhkan menggunakan regular expression."
                )

                with st.spinner("Menerapkan Data Cleaning..."):

                    def data_cleaning(text):
                        """Data Cleaning."""

                        emoticon_byte_regex = r"\s*(?:\\x[A-Fa-f0-9]{2})+"
                        url_regex = "((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+)||(http\S+))"

                        text = re.sub(
                            emoticon_byte_regex, "", text
                        )  # Remove emoticon bytes
                        text = re.sub(url_regex, "", text)  # Remove every url
                        text = re.sub(r"<[^>]*>", "", text)  # Remove html tags
                        text = re.sub(
                            r"@[A-Za-z0-9]+", "", text
                        )  # Remove twitter usernames
                        text = re.sub(r"\\n", " ", text)  # Remove every new line '\n'
                        text = re.sub("@[\w\-]+", "", text)  # Remove mentions
                        text = re.sub("RT", "", text)  # Remove every retweet symbol
                        text = re.sub("USER", "", text)  # Remove every user
                        text = re.sub(" URL", " ", text)  # Remove word URL
                        text = re.sub(" url", " ", text)  # Remove word url
                        text = re.sub("\\+", " ", text)  # Remove backslash
                        text = re.sub(
                            "\s+", " ", text
                        )  # Remove special regular expression character
                        text = re.sub("[^0-9a-zA-Z]", " ", text)  # Remove punctuation
                        text = re.sub("[^a-zA-Z]", " ", text)  # Remove numbers
                        text = re.sub(" +", " ", text)  # Remove extra spaces

                        return text

                    df["text"] = df["text"].apply(data_cleaning)
                    df.text = (
                        df.text.str.strip()
                    )  # Removes any spaces or specified characters at the start and end of a string

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("Sebelum")
                        st.write(df_raw[["raw_tweet"]].head())

                    with col2:
                        st.write("Setelah")
                        st.write(df[["text"]].head())

                st.markdown("""---""")

            if training_model.check_case_folding:
                # ------------
                # Case Folding
                # ------------
                st.subheader("Case Folding")
                st.caption("Mengubah semua huruf menjadi huruf kecil.")

                with st.spinner("Menerapkan Case Folding..."):

                    def case_folding(text):
                        """Lowercase letters."""

                        text = text.lower()
                        return text

                    df["text"] = df["text"].apply(case_folding)

                    time.sleep(2)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("Sebelum")
                        st.write(df_raw[["raw_tweet"]].head())

                    with col2:
                        st.write("Setelah")
                        st.write(df[["text"]].head())

                st.markdown("""---""")

            if training_model.check_normalization:
                # -------------
                # Normalization
                # -------------
                st.subheader("Normalization")
                st.caption("Merubah kata tidak baku menjadi baku.")

                with st.spinner("Menerapkan Normalization..."):
                    # Import Slang Dictionary from (Ibrohim & Budi, 2019)
                    alay_dict = "https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/new_kamusalay.csv"
                    alay_dict = pd.read_csv(alay_dict, encoding="latin-1", header=None)
                    alay_dict = alay_dict.rename(
                        columns={0: "original", 1: "replacement"}
                    )

                    alay_dict_map = dict(
                        zip(alay_dict["original"], alay_dict["replacement"])
                    )

                    def normalization(text):
                        """Normalization."""

                        return " ".join(
                            [
                                alay_dict_map[word] if word in alay_dict_map else word
                                for word in text.split(" ")
                            ]
                        )

                    df["text"] = df["text"].apply(normalization)

                    time.sleep(2)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("Sebelum")
                        st.write(df_raw[["raw_tweet"]].head())

                    with col2:
                        st.write("Setelah")
                        st.write(df[["text"]].head())

                st.markdown("""---""")

            if training_model.check_filtering:
                # ---------
                # Filtering
                # ---------
                st.subheader("Filtering")
                st.caption("Menghapus kata-kata yang tidak memiliki makna.")

                with st.spinner("Menerapkan Filtering..."):
                    # Initiate (Tala, F. Z., 2003) stopword
                    tala_stopword_dict = pd.read_csv(
                        "https://raw.githubusercontent.com/ryzanugrah/stopwords-id/master/stopwords-id.txt",
                        header=None,
                    )
                    tala_stopword_dict = tala_stopword_dict.rename(
                        columns={0: "stopword"}
                    )

                    # Initiate NTLK stopword
                    nltk.download("stopwords")
                    nltk.download("punkt")

                    def remove_stopword(text):
                        """Remove Stopword/Filtering."""

                        # Apply Stopword (Tala, F. Z., 2003)
                        text = " ".join(
                            [
                                ""
                                if word in tala_stopword_dict.stopword.values
                                else word
                                for word in text.split(" ")
                            ]
                        )

                        # Apply Stopword NLTK
                        nltk_stopword_dict = set(stopwords.words("indonesian"))
                        word_tokens = word_tokenize(text)
                        filtered_sentence = []
                        for word_token in word_tokens:
                            if word_token not in nltk_stopword_dict:
                                filtered_sentence.append(word_token)

                        text = " ".join(filtered_sentence)  # Join words
                        text = re.sub("  +", " ", text)  # Remove extra spaces
                        text = (
                            text.strip()
                        )  # Removes any spaces or specified characters at the start and end of a string

                        return text

                    df["text"] = df["text"].apply(remove_stopword)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("Sebelum")
                        st.write(df_raw[["raw_tweet"]].head())

                    with col2:
                        st.write("Setelah")
                        st.write(df[["text"]].head())

                st.markdown("""---""")

            if training_model.check_stemming:
                # --------
                # Stemming
                # --------
                st.subheader("Stemming")
                st.caption("Mengubah kata berimbuhan menjadi bentuk dasarnya.")

                with st.spinner("Menerapkan Stemming..."):
                    # Create stemmer
                    factory = StemmerFactory()
                    stemmer = factory.create_stemmer()

                    def stemming(text):
                        """Stemming."""

                        return stemmer.stem(text)

                    df["text"] = df["text"].apply(stemming)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("Sebelum")
                        st.write(df_raw[["raw_tweet"]].head())

                    with col2:
                        st.write("Setelah")
                        st.write(df[["text"]].head())

            st.write("")
            st.success("Text Preprocessing berhasil diterapkan.")

    return df


def index_classification(df):
    """Index Classification."""

    with st.expander("Klasifikasi indeks", expanded=True):

        with st.spinner("Menerapkan indeks pada data..."):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Indeks Data")

                def index_classification_weak(hs):
                    """Index Classification for weak hate speech."""

                    label = ""

                    if int(hs) == 1:
                        label = 1
                    else:
                        label = 0

                    return label

                def index_classification_moderate(hs):
                    """Index Classification for moderate hate speech."""

                    label = ""

                    if int(hs) == 1:
                        label = 2
                    else:
                        label = 0

                    return label

                def index_classification_strong(hs):
                    """Index Classification for strong hate speech."""

                    label = ""

                    if int(hs) == 1:
                        label = 3
                    else:
                        label = 0

                    return label

                df["label_weak"] = df["HS_Weak"].apply(index_classification_weak)
                df["label_moderate"] = df["HS_Moderate"].apply(
                    index_classification_moderate
                )
                df["label_strong"] = df["HS_Strong"].apply(index_classification_strong)

                df["label"] = df[["label_weak", "label_moderate", "label_strong"]].max(
                    1
                )
                df = df[["text", "label"]]

                df.to_csv("./data/output/dataset/dataset_preprocessed.csv", index=False)

                st.write(df.head(4))

            with col2:
                st.subheader("Jumlah Data")

                # Converting to df and assigning new name to the columns
                value_counts = df["label"].value_counts()

                df_value_counts = pd.DataFrame(value_counts)
                df_value_counts = df_value_counts.reset_index()
                df_value_counts.columns = [
                    "label",
                    "counts of label",
                ]  # Change columns name

                st.write(df_value_counts)


def split_dataset():
    """Split up dataset into train, valid, and test set."""

    with st.expander("Membagi dataset", expanded=True):
        st.subheader("Pembagian dataset")
        st.caption(
            "Membagi data dengan komposisi 80% untuk pelatihan, "
            "10% untuk validasi, dan 10% untuk pengujian."
        )

        df = pd.read_csv("./data/output/dataset/dataset_preprocessed.csv")

        with st.spinner("Membagi dataset..."):
            """
            Hold-out Validation.

            Split up dataset into train, valid, and test set
            with a ratio of 80% train, 10% test, and 10% valid.
            """

            # Define dataset
            X = df["text"]
            y = df["label"]

            # Split into 80:10:10 ration
            X_train, X_rem, y_train, y_rem = train_test_split(
                X, y, train_size=0.8, random_state=random_state
            )
            X_valid, X_test, y_valid, y_test = train_test_split(
                X_rem, y_rem, test_size=0.5, random_state=random_state
            )

            time.sleep(4)

            col1, col2, col3 = st.columns(3)

            # Describe info about train, valid, and test set
            with col1:
                st.write("Pelatihan")
                st.write(y_train.value_counts())

            with col2:
                st.write("Validasi")
                st.write(y_valid.value_counts())

            with col3:
                st.write("Pengujian")
                st.write(y_test.value_counts())

            df_train = pd.concat([X_train, y_train], axis=1)
            df_valid = pd.concat([X_valid, y_valid], axis=1)
            df_test = pd.concat([X_test, y_test], axis=1)

            df_train.to_csv("./data/output/dataset/dataset_training.csv", index=False)
            df_valid.to_csv("./data/output/dataset/dataset_validation.csv", index=False)
            df_test.to_csv("./data/output/dataset/dataset_testing.csv", index=False)

        st.markdown("""---""")

        """Balancing Data with SMOTE: Synthetic Minority Over-sampling Technique."""

        if training_model.check_smote:
            st.subheader("Data balancing dengan SMOTE")
            st.caption(
                "Menyeimbangkan jumlah data pada dataset dengan melakukan oversampling "
                "(menambah jumlah data kelas minor agar sama dengan jumlah data kelas mayor)."
            )

            with st.spinner("Menerapkan SMOTE..."):
                # Convert strings into numericals using TfidfVectorizer
                vec_train = TfidfVectorizer()
                vec_test = TfidfVectorizer()
                vec_valid = TfidfVectorizer()

                X_train = vec_train.fit_transform(df_train["text"])
                X_valid = vec_valid.fit_transform(df_valid["text"])
                X_test = vec_test.fit_transform(df_test["text"])

                sm = SMOTE(random_state=random_state)

                X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())
                X_test_res, y_test_res = sm.fit_resample(X_test, y_test.ravel())
                X_valid_res, y_valid_res = sm.fit_resample(X_valid, y_valid.ravel())

                # Convert back into strings using inverse_transform
                X_train_res = vec_train.inverse_transform(X_train_res)
                X_test_res = vec_test.inverse_transform(X_test_res)
                X_valid_res = vec_valid.inverse_transform(X_valid_res)

                pd.DataFrame({"text": X_train_res}).to_csv(
                    "./data/output/dataset/smote/X_train.csv", index=False
                )
                pd.DataFrame({"label": y_train_res}).to_csv(
                    "./data/output/dataset/smote/y_train.csv", index=False
                )

                pd.DataFrame({"text": X_test_res}).to_csv(
                    "./data/output/dataset/smote/X_test.csv", index=False
                )
                pd.DataFrame({"label": y_test_res}).to_csv(
                    "./data/output/dataset/smote/y_test.csv", index=False
                )

                pd.DataFrame({"text": X_valid_res}).to_csv(
                    "./data/output/dataset/smote/X_valid.csv", index=False
                )
                pd.DataFrame({"label": y_valid_res}).to_csv(
                    "./data/output/dataset/smote/y_valid.csv", index=False
                )

                time.sleep(5)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("Pelatihan")
                st.write("Jumlah label '0': {}".format(sum(y_train_res == 0)))
                st.write("Jumlah label '1': {}".format(sum(y_train_res == 1)))
                st.write("Jumlah label '2': {}".format(sum(y_train_res == 2)))
                st.write("Jumlah label '3': {}".format(sum(y_train_res == 3)))
                st.write("Total: {}".format(len(y_train_res)))

            with col2:
                st.write("Validasi")
                st.write("Jumlah label '0': {}".format(sum(y_valid_res == 0)))
                st.write("Jumlah label '1': {}".format(sum(y_valid_res == 1)))
                st.write("Jumlah label '2': {}".format(sum(y_valid_res == 2)))
                st.write("Jumlah label '3': {}".format(sum(y_valid_res == 3)))
                st.write("Total: {}".format(len(y_valid_res)))

            with col3:
                st.write("Pengujian")
                st.write("Jumlah label '0': {}".format(sum(y_test_res == 0)))
                st.write("Jumlah label '1': {}".format(sum(y_test_res == 1)))
                st.write("Jumlah label '2': {}".format(sum(y_test_res == 2)))
                st.write("Jumlah label '3': {}".format(sum(y_test_res == 3)))
                st.write("Total: {}".format(len(y_test_res)))

            # Clean the dataset to make sure data is clean
            def data_cleaning(text):
                text = re.sub("['']", "", text)
                text = text.replace("[", "").replace("]", "")
                return text

            df_train = pd.read_csv("./data/output/dataset/smote/X_train.csv")
            df_valid = pd.read_csv("./data/output/dataset/smote/X_valid.csv")
            df_test = pd.read_csv("./data/output/dataset/smote/X_test.csv")

            df_train["text"] = df_train["text"].apply(data_cleaning)
            df_train.text = df_train.text.str.strip()

            df_valid["text"] = df_valid["text"].apply(data_cleaning)
            df_valid.text = df_valid.text.str.strip()

            df_test["text"] = df_test["text"].apply(data_cleaning)
            df_test.text = df_test.text.str.strip()

            df_train.to_csv("./data/output/dataset/smote/X_train.csv", index=False)
            df_valid.to_csv("./data/output/dataset/smote/X_valid.csv", index=False)
            df_test.to_csv("./data/output/dataset/smote/X_test.csv", index=False)

            # Save oversampled data training
            df_X_train = pd.read_csv("./data/output/dataset/smote/X_train.csv")
            df_y_train = pd.read_csv("./data/output/dataset/smote/y_train.csv")
            df_train = pd.concat([df_X_train, df_y_train], axis=1)

            # Save oversampled data validation
            df_X_valid = pd.read_csv("./data/output/dataset/smote/X_valid.csv")
            df_y_valid = pd.read_csv("./data/output/dataset/smote/y_valid.csv")
            df_valid = pd.concat([df_X_valid, df_y_valid], axis=1)

            # Save oversampled data testing
            df_X_test = pd.read_csv("./data/output/dataset/smote/X_test.csv")
            df_y_test = pd.read_csv("./data/output/dataset/smote/y_test.csv")
            df_test = pd.concat([df_X_test, df_y_test], axis=1)

            st.markdown("""---""")

        st.subheader("Finalisasi dataset")
        st.caption(
            "Jumlah dataset setelah menghapus nilai data yang hilang pada dataset."
        )

        with st.spinner("Membersihkan data..."):
            """Label Classification."""

            # Data training
            def label_classification(hs):
                label = ""

                if int(hs) == 1:
                    label = "HS_Weak"
                elif int(hs) == 2:
                    label = "HS_Moderate"
                elif int(hs) == 3:
                    label = "HS_Strong"
                else:
                    label = "Non_HS"

                return label

            df_train["label"] = df_train["label"].apply(label_classification)
            df_train = df_train[["text", "label"]]
            df_train.to_csv("./data/output/dataset/dataset_training.csv", index=False)

            # Data validation
            def label_classification(hs):
                label = ""

                if int(hs) == 1:
                    label = "HS_Weak"
                elif int(hs) == 2:
                    label = "HS_Moderate"
                elif int(hs) == 3:
                    label = "HS_Strong"
                else:
                    label = "Non_HS"

                return label

            df_valid["label"] = df_valid["label"].apply(label_classification)
            df_valid = df_valid[["text", "label"]]
            df_valid.to_csv("./data/output/dataset/dataset_validation.csv", index=False)

            # Data testing
            def label_classification(hs):
                label = ""

                if int(hs) == 1:
                    label = "HS_Weak"
                elif int(hs) == 2:
                    label = "HS_Moderate"
                elif int(hs) == 3:
                    label = "HS_Strong"
                else:
                    label = "Non_HS"

                return label

            df_test["label"] = df_test["label"].apply(label_classification)
            df_test = df_test[["text", "label"]]
            df_test.to_csv("./data/output/dataset/dataset_testing.csv", index=False)

        # Remove missing value if any, just for make sure once again
        # Before
        df_train = pd.read_csv("./data/output/dataset/dataset_training.csv")
        df_train.isnull().sum()
        df_test = pd.read_csv("./data/output/dataset/dataset_testing.csv")
        df_test.isnull().sum()
        df_valid = pd.read_csv("./data/output/dataset/dataset_validation.csv")
        df_valid.isnull().sum()

        # After
        df_train = df_train[df_train["text"].notna()]
        df_train.isnull().sum()
        df_test = df_test[df_test["text"].notna()]
        df_test.isnull().sum()
        df_valid = df_valid[df_valid["text"].notna()]
        df_valid.isnull().sum()

        col1, col2 = st.columns(2)

        with col1:
            st.write("Jumlah Pelatihan Data: ", len(df_train))
            st.write("Jumlah Validasi Data: ", len(df_valid))
            st.write("Jumlah Pengujian Data: ", len(df_test))

            df_train.to_csv("./data/output/dataset/dataset_training.csv", index=False)
            df_test.to_csv("./data/output/dataset/dataset_testing.csv", index=False)
            df_valid.to_csv("./data/output/dataset/dataset_validation.csv", index=False)

        with col2:
            st.markdown(
                get_table_download_link(
                    df_train, "dataset_training.csv", "Unduh dataset pelatihan"
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                get_table_download_link(
                    df_valid,
                    "dataset_validation.csv",
                    "Unduh dataset validasi",
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                get_table_download_link(
                    df_test, "dataset_testing.csv", "Unduh dataset pengujian"
                ),
                unsafe_allow_html=True,
            )


def fine_tuning():
    """
    Modeling Pretrained Model (Training Model).
    """

    global i2w, model, tokenizer, testing_loader

    with st.expander("Memuat pre-trained model", expanded=True):
        """Load tokenizer, config, and IndoBERT pre-trained model."""

        st.subheader("Memuat tokenizer, config, dan IndoBERT pre-trained model")

        # Load tokenizer
        with st.spinner("Memuat BERT tokenizer..."):
            tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

        # Load config
        with st.spinner("Memuat BERT config..."):
            config = BertConfig.from_pretrained(
                "indobenchmark/indobert-base-p1",
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                hidden_dropout_prob=hidden_dropout_prob,
                num_labels=num_labels,
            )
            config.num_labels = MultiLabelHateSpeechClassificationDataset.NUM_LABELS

        # Instantiate model
        with st.spinner("Mengunduh BERT pre-trained model..."):
            model = BertForSequenceClassification.from_pretrained(
                "indobenchmark/indobert-base-p1", config=config
            )

        # Tell pytorch to run this model on the GPU
        model.cuda()

        st.success("Memuat pre-trained model berhasil.")

    with st.expander("Menerapkan tokenizer pada contoh kalimat", expanded=True):
        """Apply the tokenizer to one sentence just to see the output"""

        st.subheader("Menerapkan tokenizer")
        st.caption(
            "Proses memisahkan setiap kata pada suatu kalimat menjadi token "
            "agar dapat dibaca oleh BERT."
        )

        text_1 = "kebahagiaan terbesarku adalah melihatmu bersama dengan dirinya"
        text_2 = "dengan begitu kau akan tenang bersamanya"

        text = (text_1, text_2)

        text = " ".join(text)

        token = tokenizer.tokenize(text)  # Tokenizing
        encoding = tokenizer.encode(text_1, text_2)  # Token ids
        decoding = tokenizer.decode(encoding)  # Token embeddings
        encoding_input = tokenizer(text)

        st.write("Text: ", text)
        st.write("Tokenized: ", token)
        st.write("Token Embeddings: ", decoding)
        st.write("Token IDs: ", encoding_input)

    with st.expander("Membuat DataLoader", expanded=True):
        """
        Create DataLoader.

        Create an iterator for dataset using Torch's DataLoader class.
        This helps save memory during training and don't have to load the entire dataset into memory.
        """

        st.subheader("Membuat DataLoader")
        st.caption(
            "DataLoader membantu menghemat memori selama proses pelatihan "
            "agar perangkat tidak perlu memuat seluruh dataset ke dalam memori."
        )

        with st.spinner("Membuat DataLoader..."):
            time.sleep(4)

        training_dataset_path = "./data/output/dataset/dataset_training.csv"
        validation_dataset_path = "./data/output/dataset/dataset_validation.csv"
        testing_dataset_path = "./data/output/dataset/dataset_testing.csv"

        training_dataset = MultiLabelHateSpeechClassificationDataset(
            training_dataset_path, tokenizer, lowercase=True
        )
        validation_dataset = MultiLabelHateSpeechClassificationDataset(
            validation_dataset_path, tokenizer, lowercase=True
        )
        testing_dataset = MultiLabelHateSpeechClassificationDataset(
            testing_dataset_path, tokenizer, lowercase=True
        )

        training_loader = HateSpeechClassificationDataLoader(
            dataset=training_dataset,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
        validation_loader = HateSpeechClassificationDataLoader(
            dataset=validation_dataset,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        testing_loader = HateSpeechClassificationDataLoader(
            dataset=testing_dataset,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

        st.info(
            "\nModel akan dilatih dengan "
            + str(len(training_dataset))
            + " pelatihan data, "
            + str(len(validation_dataset))
            + " validasi data, dan "
            + str(len(testing_dataset))
            + " pengujian data."
            + "\n\nModel akan dilatih dengan "
            + str(len(training_loader))
            + " pelatihan dataloader, "
            + str(len(validation_loader))
            + " validasi dataloader, dan "
            + str(len(testing_loader))
            + " pengujian dataloader."
        )

        with st.container():
            """Labeling the index and vice versa"""

            w2i, i2w = (
                MultiLabelHateSpeechClassificationDataset.LABEL2INDEX,
                MultiLabelHateSpeechClassificationDataset.INDEX2LABEL,
            )

            st.write("Klasifikasi Label dan Indeks")
            st.write(w2i)
            st.write(i2w)

    with st.expander(
        "Klasifikasi Teks/Kalimat dengan model IndoBERT pre-trained",
        expanded=True,
    ):
        """Classification only with IndoBERT pre-trained model."""

        st.subheader("Klasifikasi Kalimat Hate Speech Bahasa Indonesia")
        st.caption(
            "Klasifikasi beberapa kalimat bahasa Indonesia "
            "beserta persentase kemungkinannya."
        )

        with st.spinner("Menerapkan klasifikasi dengan model IndoBERT pre-trained..."):
            time.sleep(3)

        with st.container():
            text = "kebahagiaan terbesarku adalah melihatmu bersama dengan dirinya.."
            subwords = tokenizer.encode(text)
            subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

            logits = model(subwords)[0]
            label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()

            st.write(f"Text: {text}")
            st.write(
                f"Label: {i2w[label]} ({F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}%)"
            )

            st.write("")

            text = "Astaghfirullah, MONSTER berhijab ? Kadrun nih pasti. frustasi karena pak @jokowi menang Pilpres lagi ?"
            subwords = tokenizer.encode(text)
            subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

            logits = model(subwords)[0]
            label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()

            st.write(f"Text: {text}")
            st.write(
                f"Label: {i2w[label]} ({F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}%)"
            )

            st.write("")

            text = "Semua orang #indocina adalah kecoa-kecoa busuk pencari untung dan harus diusir jauh-jauh!!"
            subwords = tokenizer.encode(text)
            subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

            logits = model(subwords)[0]
            label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()

            st.write(f"Text: {text}")
            st.write(
                f"Label: {i2w[label]} ({F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}%)"
            )

    with st.expander("Fine-tuning model", expanded=True):
        """
        Fine-Tuning.

        Now the input data is properly formatted, it's time to fine-tune the BERT model.
        """

        with st.container():
            st.subheader("Optimizer")
            st.caption(
                "Optimizer digunakan untuk mengoptimalkan performa model berdasarkan "
                "fungsi loss dan parameter yang sudah ada."
            )

            with st.spinner("Menerapkan AdamW Optimizer..."):
                # Apply AdamW Optimizer
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay
                )
                model = model.cuda()

                time.sleep(2)

                st.success("AdamW Optimizer berhasil diterapkan.")

        st.markdown("""---""")

        with st.container():
            """
            This training code is based on the `run_glue.py` script here:
            https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
            """

            st.subheader("Fine-Tuning")
            st.caption(
                "Proses memuat model BERT yang telah dilatih untuk dikonfigurasi "
                "agar membuatnya dapat melakukan tugas serupa lainnya."
            )

            start_training = timeit.default_timer()

            train_acc_lists = []
            train_pre_lists = []
            train_rec_lists = []
            train_f1_lists = []

            eval_acc_lists = []
            eval_pre_lists = []
            eval_rec_lists = []
            eval_f1_lists = []

            train_loss_lists = []
            eval_loss_lists = []

            # Fine-tuning
            for epoch in range(0, epochs):

                # ========================================
                #               Training
                # ========================================

                model.train()
                torch.set_grad_enabled(True)

                # Reset the total loss for this epoch
                total_train_loss = 0

                list_hyp, list_label = [], []

                train_pbar = stqdm(
                    training_loader, leave=True, total=len(training_loader)
                )
                for i, batch_data in enumerate(train_pbar):
                    # Forward model
                    (loss, batch_hyp, batch_label,) = forward_sequence_classification(
                        model, batch_data[:-1], i2w=i2w, device="cuda"
                    )

                    # Update model
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    tr_loss = loss.item()
                    total_train_loss = total_train_loss + tr_loss

                    # Calculate metrics
                    list_hyp += batch_hyp
                    list_label += batch_label

                    train_pbar.set_description(
                        "(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format(
                            (epoch + 1),
                            total_train_loss / (i + 1),
                            get_lr(optimizer),
                        )
                    )

                # Calculate train metric
                metrics = multi_label_hate_speech_classification_metrics_fn(
                    list_hyp, list_label
                )
                st.write(
                    "(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format(
                        (epoch + 1),
                        total_train_loss / (i + 1),
                        metrics_to_string(metrics),
                        get_lr(optimizer),
                    )
                )

                train_acc_lists.append(metrics["ACC"])
                train_pre_lists.append(metrics["PRE"])
                train_rec_lists.append(metrics["REC"])
                train_f1_lists.append(metrics["F1"])
                current_train_loss = round(total_train_loss / (i + 1), 4)
                train_loss_lists.append(current_train_loss)

                # ========================================
                #               Validation
                # ========================================

                """
                After the completion of each training epoch, measure our performance on
                our validation set.
                """

                model.eval()
                torch.set_grad_enabled(False)

                total_loss, total_correct, total_labels = 0, 0, 0
                list_hyp, list_label = [], []

                pbar = stqdm(
                    validation_loader, leave=True, total=len(validation_loader)
                )

                for i, batch_data in enumerate(pbar):
                    batch_seq = batch_data[-1]
                    (loss, batch_hyp, batch_label,) = forward_sequence_classification(
                        model, batch_data[:-1], i2w=i2w, device="cuda"
                    )

                    # Calculate total loss
                    valid_loss = loss.item()
                    total_loss = total_loss + valid_loss

                    # Calculate evaluation metrics
                    list_hyp += batch_hyp
                    list_label += batch_label
                    metrics = multi_label_hate_speech_classification_metrics_fn(
                        list_hyp, list_label
                    )

                    pbar.set_description(
                        "(Epoch {}) VALID LOSS:{:.4f} {}".format(
                            (epoch + 1),
                            total_loss / (i + 1),
                            metrics_to_string(metrics),
                        )
                    )

                metrics = multi_label_hate_speech_classification_metrics_fn(
                    list_hyp, list_label
                )
                st.write(
                    "(Epoch {}) VALID LOSS:{:.4f} {}".format(
                        (epoch + 1),
                        total_loss / (i + 1),
                        metrics_to_string(metrics),
                    )
                )

                eval_acc_lists.append(metrics["ACC"])
                eval_pre_lists.append(metrics["PRE"])
                eval_rec_lists.append(metrics["REC"])
                eval_f1_lists.append(metrics["F1"])
                current_eval_loss = round(total_loss / (i + 1), 4)
                eval_loss_lists.append(current_eval_loss)

            # Show the elapsed time of fine-tuning.
            stop_training = timeit.default_timer()
            training_time = int(round((stop_training - start_training)))
            mins_training = str((training_time % 3600) // 60)
            secs_training = str((training_time % 3600) % 60)
            time_training = (
                "Pelatihan model selesai dalam waktu {} menit dan {} detik.".format(
                    mins_training, secs_training
                )
            )

            st.success(time_training)

        st.markdown("""---""")

        with st.container():
            st.subheader("Diagram Hasil Pelatihan")

            # Show bar plot for Training and Validation Accuracy, Precision, Recall, and F1-score
            fig, ax = plt.subplots(figsize=(12, 8))

            if epochs == 2:
                epoch = [1, 2]
            elif epochs == 3:
                epoch = [1, 2, 3]
            elif epochs == 4:
                epoch = [1, 2, 3, 4]

            width = 0.25

            col1, col2 = st.columns(2)

            with col1:
                with st.container():
                    st.write("Accuracy")

                    # Show accuracy bar plot
                    fig, ax = plt.subplots(figsize=(12, 8))

                    # Set position of bar on X axis
                    br1 = np.arange(len(train_acc_lists)) / 1.5
                    br2 = [x + width for x in br1]

                    plt.bar(br1, train_acc_lists, label="train", width=width)
                    plt.bar(br2, eval_acc_lists, label="valid", width=width)

                    # Annotate
                    for p in ax.patches:
                        ax.annotate(
                            np.round(p.get_height(), decimals=2),
                            (p.get_x() + p.get_width() / 2.0, p.get_height()),
                            ha="center",
                            va="center",
                            xytext=(0, 10),
                            textcoords="offset points",
                        )

                    plt.xlabel("Epoch")
                    plt.ylabel("Accuracy")
                    plt.title("Training and Validation Accuracy")
                    plt.xticks(
                        [r / 1.5 - 0.13 + width for r in range(len(train_acc_lists))],
                        epoch,
                    )
                    plt.legend()

                    st.pyplot(fig)

                with st.container():
                    st.write("Recall")

                    # Show recall bar plot
                    fig, ax = plt.subplots(figsize=(12, 8))

                    # Set position of bar on X axis
                    br1 = np.arange(len(train_rec_lists)) / 1.5
                    br2 = [x + width for x in br1]

                    plt.bar(br1, train_rec_lists, label="train", width=width)
                    plt.bar(br2, eval_rec_lists, label="valid", width=width)

                    # Annotate
                    for p in ax.patches:
                        ax.annotate(
                            np.round(p.get_height(), decimals=2),
                            (p.get_x() + p.get_width() / 2.0, p.get_height()),
                            ha="center",
                            va="center",
                            xytext=(0, 10),
                            textcoords="offset points",
                        )

                    plt.xlabel("Epoch")
                    plt.ylabel("Recall")
                    plt.title("Training and Validation Recall")
                    plt.xticks(
                        [r / 1.5 - 0.13 + width for r in range(len(train_rec_lists))],
                        epoch,
                    )
                    plt.legend()

                    st.pyplot(fig)

            with col2:
                with st.container():
                    st.write("Precision")

                    # Show precision bar plot
                    fig, ax = plt.subplots(figsize=(12, 8))

                    # Set position of bar on X axis
                    br1 = np.arange(len(train_pre_lists)) / 1.5
                    br2 = [x + width for x in br1]

                    plt.bar(br1, train_pre_lists, label="train", width=width)
                    plt.bar(br2, eval_pre_lists, label="valid", width=width)

                    # Annotate
                    for p in ax.patches:
                        ax.annotate(
                            np.round(p.get_height(), decimals=2),
                            (p.get_x() + p.get_width() / 2.0, p.get_height()),
                            ha="center",
                            va="center",
                            xytext=(0, 10),
                            textcoords="offset points",
                        )

                    plt.xlabel("Epoch")
                    plt.ylabel("Precision")
                    plt.title("Training and Validation Precision")
                    plt.xticks(
                        [r / 1.5 - 0.13 + width for r in range(len(train_pre_lists))],
                        epoch,
                    )
                    plt.legend()

                    st.pyplot(fig)

                with st.container():
                    st.write("F1-score")

                    # Show f1-score bar plot
                    fig, ax = plt.subplots(figsize=(12, 8))

                    # Set position of bar on X axis
                    br1 = np.arange(len(train_f1_lists)) / 1.5
                    br2 = [x + width for x in br1]

                    plt.bar(br1, train_f1_lists, label="train", width=width)
                    plt.bar(br2, eval_f1_lists, label="valid", width=width)

                    # Annotate
                    for p in ax.patches:
                        ax.annotate(
                            np.round(p.get_height(), decimals=2),
                            (p.get_x() + p.get_width() / 2.0, p.get_height()),
                            ha="center",
                            va="center",
                            xytext=(0, 10),
                            textcoords="offset points",
                        )

                    plt.xlabel("Epoch")
                    plt.ylabel("F1-score")
                    plt.title("Training and Validation F1-score")
                    plt.xticks(
                        [r / 1.5 - 0.13 + width for r in range(len(train_f1_lists))],
                        epoch,
                    )
                    plt.legend()

                    st.pyplot(fig)

            with st.container():
                st.write("Loss")

                # Show loss bar plot
                fig, ax = plt.subplots(figsize=(12, 8))

                # Set position of bar on X axis
                br1 = np.arange(len(train_loss_lists)) / 1.5
                br2 = [x + width for x in br1]

                plt.bar(br1, train_loss_lists, label="train", width=width)
                plt.bar(br2, eval_loss_lists, label="valid", width=width)

                # Annotate
                for p in ax.patches:
                    ax.annotate(
                        np.round(p.get_height(), decimals=2),
                        (p.get_x() + p.get_width() / 2.0, p.get_height()),
                        ha="center",
                        va="center",
                        xytext=(0, 10),
                        textcoords="offset points",
                    )

                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Training and Validation Loss")
                plt.xticks(
                    [r / 1.5 - 0.13 + width for r in range(len(train_loss_lists))],
                    epoch,
                )
                plt.legend()

                st.pyplot(fig)

        st.markdown("""---""")

        with st.container():
            # Constructing dataframe from training
            df_stats = {
                "Train Accuracy": np.round(train_acc_lists, 2),
                "Train Precision": np.round(train_pre_lists, 2),
                "Train Recall": np.round(train_rec_lists, 2),
                "Train F1": np.round(train_f1_lists, 2),
                "Valid Accuracy": np.round(eval_acc_lists, 2),
                "Valid Precision": np.round(eval_pre_lists, 2),
                "Valid Recall": np.round(eval_rec_lists, 2),
                "Valid F1": np.round(eval_f1_lists, 2),
                "Train Loss": np.round(train_loss_lists, 2),
                "Valid Loss": np.round(eval_loss_lists, 2),
            }

            # Create a dataframe from training
            df_stats = pd.DataFrame(data=df_stats)

            # Rename row index as 'Epoch'
            df_stats = df_stats.rename_axis("Epoch")

            # Display the table
            df_stats.index = df_stats.index + 1
            st.subheader("Hasil Pelatihan")
            st.dataframe(df_stats)
            st.markdown(
                get_table_download_link(df_stats, "training_result.csv", "Unduh tabel"),
                unsafe_allow_html=True,
            )


def evaluate_model():
    """
    Evaluate on Test Set.
    With the test set prepared, we can apply our fine-tuned model to generate predictions on the test set.
    """

    with st.expander("Evaluasi hasil pelatihan model", expanded=True):
        with st.container():
            st.subheader("Memuat DataLoader dan List Prediksi")

            # Put model in evaluation mode
            model.eval()
            torch.set_grad_enabled(False)

            total_loss, total_correct, total_labels = 0, 0, 0
            list_hyp, list_label = [], []

            pbar = stqdm(testing_loader, leave=True, total=len(testing_loader))

            for i, batch_data in enumerate(pbar):
                _, batch_hyp, _ = forward_sequence_classification(
                    model, batch_data[:-1], i2w=i2w, device="cuda"
                )
                list_hyp += batch_hyp

            st.success("DataLoader pengujian berhasil diterapkan.")

            st.markdown("""---""")

            prediction = pd.DataFrame({"label": list_hyp}).reset_index()

            st.subheader("Tabel Prediksi")
            st.write(prediction)

        with st.spinner("Mengevaluasi..."):
            prediction_list = []

            for i in prediction["label"]:
                if i == "HS_Weak":
                    prediction_list.append(1)
                elif i == "HS_Moderate":
                    prediction_list.append(2)
                elif i == "HS_Strong":
                    prediction_list.append(3)
                else:
                    prediction_list.append(0)

            data_test = pd.read_csv("./data/output/dataset/dataset_testing.csv")

            def label_classification(hs):
                label = ""

                if str(hs) == "HS_Weak":
                    label = 1
                elif str(hs) == "HS_Moderate":
                    label = 2
                elif str(hs) == "HS_Strong":
                    label = 3
                else:
                    label = 0

                return label

            data_test["label"] = data_test["label"].apply(label_classification)
            data_test = data_test[["text", "label"]]

        st.markdown("""---""")

        with st.container():
            st.subheader("Hasil Klasifikasi")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Accuracy",
                    value=f"{round(accuracy_score(data_test['label'], prediction_list) * 100)}%",
                )

                f1_score_micro = (
                    f1_score(data_test["label"], prediction_list, average="micro") * 100
                )
                st.metric("F1-score (Micro Avg)", value=f"{round(f1_score_micro)}%")

            with col2:
                st.metric(
                    "Precision",
                    value=f"{round(precision_score(data_test['label'], prediction_list, average='macro') * 100)}%",
                )

                f1_score_macro = (
                    f1_score(data_test["label"], prediction_list, average="macro") * 100
                )
                st.metric("F1-score (Macro Avg)", value=f"{round(f1_score_macro)}%")

            with col3:
                st.metric(
                    "Recall",
                    value=f"{round(recall_score(data_test['label'], prediction_list, average='macro') * 100)}%",
                )

                f1_score_weighted = (
                    f1_score(data_test["label"], prediction_list, average="weighted")
                    * 100
                )
                st.metric(
                    "F1-score (Weighted Avg)", value=f"{round(f1_score_weighted)}%"
                )

        st.markdown("""---""")

        with st.container():
            # Classification Report
            st.subheader("Classification Report")
            st.text(classification_report(data_test["label"], prediction_list))

            report = classification_report(
                data_test["label"], prediction_list, output_dict=True
            )
            report = pd.DataFrame(report).round(2).transpose()

            st.write("")
            st.write("Tabel Classification Report")
            st.dataframe(report)
            st.markdown(
                get_table_download_link(
                    report, "classification_report.csv", "Unduh tabel"
                ),
                unsafe_allow_html=True,
            )

        st.markdown("""---""")

        with st.container():
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(
                confusion_matrix(data_test["label"], prediction_list),
                annot=True,
                fmt=".0f",
                ax=ax,
            )
            plt.xlabel("True Label")
            plt.ylabel("Predicted Label")

            # Show confusion matrix
            st.pyplot(fig)
            st.markdown(
                get_plt_image_download_link(
                    fig, "confusion_matrix.png", "Unduh Confusion Matrix"
                ),
                unsafe_allow_html=True,
            )

    with st.expander("Perbandingan hasil label asli dan prediksi", expanded=True):
        """
        Compare True and Prediction Result.
        True and prediction result should at least have the similar or same result.
        """

        st.subheader("Jumlah Label")

        with st.container():
            # True label
            amount_of_true_non_hs = 0
            amount_of_true_hs_weak = 0
            amount_of_true_hs_moderate = 0
            amount_of_true_hs_strong = 0

            for i in data_test["label"]:
                if i == 1:
                    amount_of_true_hs_weak += 1
                elif i == 2:
                    amount_of_true_hs_moderate += 1
                elif i == 3:
                    amount_of_true_hs_strong += 1
                else:
                    amount_of_true_non_hs += 1

            # Prediction label
            amount_of_pred_non_hs = 0
            amount_of_pred_hs_weak = 0
            amount_of_pred_hs_moderate = 0
            amount_of_pred_hs_strong = 0

            for i in prediction_list:
                if i == 1:
                    amount_of_pred_hs_weak += 1
                elif i == 2:
                    amount_of_pred_hs_moderate += 1
                elif i == 3:
                    amount_of_pred_hs_strong += 1
                else:
                    amount_of_pred_non_hs += 1

            st.write(
                "Jumlah Non-Hate Speech: Asli {} dan Prediksi {}".format(
                    amount_of_true_non_hs, amount_of_pred_non_hs
                )
            )
            st.write(
                "Jumlah Hate Speech Lemah: Asli {} dan Prediksi {}".format(
                    amount_of_true_hs_weak, amount_of_pred_hs_weak
                )
            )
            st.write(
                "Jumlah Hate Speech Menengah: Asli {} dan Prediksi {}".format(
                    amount_of_true_hs_moderate, amount_of_pred_hs_moderate
                )
            )
            st.write(
                "Jumlah Hate Speech Kuat: Asli {} dan Prediksi {}".format(
                    amount_of_true_hs_strong, amount_of_pred_hs_strong
                )
            )

        st.markdown("""---""")

        with st.container():
            st.subheader("Diagram Perbandingan Label")
            st.caption("Diagram perbandingan label asli dengan label prediksi.")

            with st.spinner("Menampilkan Diagram perbandingan..."):
                """Show compare result on chart."""

                fig, (ax1, ax2) = plt.subplots(
                    1, 2, figsize=(20, 10), subplot_kw=dict(aspect="equal")
                )
                labels = "Non_HS", "HS_Weak", "HS_Moderate", "HS_Strong"
                labels = [x.split()[-1] for x in labels]

                # True result
                sizes = [
                    amount_of_true_non_hs,
                    amount_of_true_hs_weak,
                    amount_of_true_hs_moderate,
                    amount_of_true_hs_strong,
                ]
                explode = (0.1, 0, 0, 0)

                ax1.pie(
                    sizes,
                    explode=explode,
                    labels=labels,
                    autopct="%1.1f%%",
                    shadow=True,
                    startangle=90,
                    textprops=dict(color="white", weight="bold"),
                )
                ax1.axis("equal")
                ax1.set_title("Labeling True Result")

                # Prediction result
                sizes = [
                    amount_of_pred_non_hs,
                    amount_of_pred_hs_weak,
                    amount_of_pred_hs_moderate,
                    amount_of_pred_hs_strong,
                ]
                explode = (0.1, 0, 0, 0)

                ax2.pie(
                    sizes,
                    explode=explode,
                    labels=labels,
                    autopct="%1.1f%%",
                    shadow=True,
                    startangle=90,
                    textprops=dict(color="white", weight="bold"),
                )
                ax2.axis("equal")
                ax2.set_title("Labeling Prediction Result")

                # Set legend label
                ax1.legend(
                    labels,
                    title="Label",
                    loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1),
                )

                ax1.plot()
                ax2.plot()

                time.sleep(3)

            st.pyplot(fig)
            st.markdown(
                get_plt_image_download_link(
                    fig, "prediction_result.png", "Unduh hasil prediksi"
                ),
                unsafe_allow_html=True,
            )

    with st.expander(
        "Klasifikasi Teks/Kalimat dengan model yang telah dilatih", expanded=True
    ):
        """Test Fine-Tuned Model."""

        st.subheader("Uji Coba Model")
        st.caption(
            "Klasifikasi beberapa kalimat bahasa Indonesia "
            "beserta persentase kemungkinannya."
        )

        with st.spinner("Menerapkan klasifikasi..."):
            time.sleep(3)

        with st.container():
            text = "kebahagiaan terbesarku adalah melihatmu bersama dengan dirinya.."
            subwords = tokenizer.encode(text)
            subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

            logits = model(subwords)[0]
            label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()

            st.write(f"Text: {text}")
            st.write(
                f"Label: {i2w[label]} ({F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}%)"
            )

            st.write("")

            text = "Astaghfirullah, MONSTER berhijab ? Kadrun nih pasti. frustasi karena pak @jokowi menang Pilpres lagi ?"
            subwords = tokenizer.encode(text)
            subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

            logits = model(subwords)[0]
            label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()

            st.write(f"Text: {text}")
            st.write(
                f"Label: {i2w[label]} ({F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}%)"
            )

            st.write("")

            text = "Semua orang #indocina adalah kecoa-kecoa busuk pencari untung dan harus diusir jauh-jauh!!"
            subwords = tokenizer.encode(text)
            subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

            logits = model(subwords)[0]
            label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()

            st.write(f"Text: {text}")
            st.write(
                f"Label: {i2w[label]} ({F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}%)"
            )

            st.write("")

            text = "kemaren gue ga di ajak tai emang"
            subwords = tokenizer.encode(text)
            subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

            logits = model(subwords)[0]
            label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()

            st.write(f"Text: {text}")
            st.write(
                f"Label: {i2w[label]} ({F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}%)"
            )


def elapsed_time(execution_time):
    """
    Show the elapsed time of training proccess.
    """

    mins = str((execution_time % 3600) // 60)
    secs = str((execution_time % 3600) % 60)
    process_time = "Proses pelatihan dan pengujian selesai dalam waktu {} menit dan {} detik.".format(
        mins, secs
    )

    return st.success(process_time)
