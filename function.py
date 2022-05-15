import pandas as pd
import base64
import tweepy as tw
import re
import numpy as np
import string
import pickle
import torch
from torch import optim
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
import streamlit as st
import io
import os
import json
import uuid
import gdown
from pyunpack import Archive

path = "models/tokenizer.pkl"
# Load trained tokenizer with pickle
with open(path, "rb") as handle:
    tokenizer = pickle.load(handle)

# Instantiate trained
config = BertConfig.from_pretrained("models/config.json")

# url = "https://drive.google.com/"
output = "models/pytorch_model.bin"
# gdown.download(url, output, quiet=False)
model = BertForSequenceClassification.from_pretrained(output, config=config)

# Load Kamus Alay untuk normalisasi
alay_dict = "https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/new_kamusalay.csv"
alay_dict = pd.read_csv(alay_dict, encoding="latin-1", header=None)
alay_dict = alay_dict.rename(columns={0: "original", 1: "replacement"})


# ----------------------------------------------
# DEFINE FUNCTIONS
# ----------------------------------------------

# Function 1
# -----------------
def get_table_download_link(df):
    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub("\d+", "", button_uuid)
    custom_css = f"""
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }}
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    href = (
        custom_css
        + f'<a href="data:file/csv;base64,{b64}" download="tweets.csv" id="{button_id}">CLICK HERE TO DOWNLOAD CSV FILE</a></br></br>'
    )
    return href


# Function 2:
# ----------------
# Hit twitter api & add basic features & output 2 dataframes
# @st.cache(suppress_st_warning=True,allow_output_mutation=True)
def twitter_get(select_hashtag_keyword, user_word_entry, num_of_tweets):

    # Set up Twitter API access
    # Define access keys and tokens
    consumer_key = st.secrets["consumer_key"]
    consumer_secret = st.secrets["consumer_secret"]
    access_token = st.secrets["access_token"]
    access_token_secret = st.secrets["access_token_secret"]

    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)

    # Keyword or hashtag
    if select_hashtag_keyword == "Hashtag":
        user_word = "#" + user_word_entry
    else:
        user_word = user_word_entry

    # Retweets (assumes yes)
    user_word = user_word + " -filter:retweets"
    tweets = tw.Cursor(api.search_tweets, q=user_word, tweet_mode="extended", lang="id").items(
        num_of_tweets
    )

    # Store as dataframe
    tweet_metadata = [
        [tweet.created_at, tweet.id, tweet.user.screen_name, tweet.full_text]
        for tweet in tweets
    ]
    df_tweets = pd.DataFrame(
        data=tweet_metadata, columns=["created_at", "id", "user", "full_text"]
    )

    df_tweets["created_dttime"] = df_tweets["created_at"].apply(
        lambda x: x.strftime("%a %b %d %Y %X")
    )
    # Create a new text variable to do manipulations on
    df_tweets["clean_text"] = df_tweets.full_text

    df_new = df_tweets[["created_at", "user", "full_text"]]
    df_new = df_new.rename(columns={"user": "Username", "full_text": "Tweet"})

    return df_tweets, df_new


def preprocessing(text):
    url_regex = "((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+)||(http\S+))"
    emoticon_byte_regex = r"\s*(?:\\x[A-Fa-f0-9]{2})+"
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r"", text)  # remove emoji
    text = (
        " " + text
    )  # added space because there was some weirdness for first word (strip later)
    text = re.sub(r"(\s)@\w+", "", text)  # remove whole word if starts with @
    text = re.sub(r"(\s).@\w+", "", text)  # remove whole word if starts with @
    text = re.sub(r"\\n", " ", text)  # Remove every '\n'
    text = re.sub(
        r"(\s)\w*\d\w*\w+", "", text
    )  # remove whole word if starts with number
    text = re.sub(r"https\:\/\/t\.co\/*\w*", "", text)  # remove https links
    text = re.sub(
        "[%s]" % re.escape(string.punctuation), "", text
    )  # removes punctuation
    text = re.sub("\[.*?\]", "", text)  # removes text in square brackets
    text = re.sub("\\+", " ", text)
    text = re.sub(emoticon_byte_regex, "", text)  # Remove emoticon bytes
    text = re.sub("[^0-9a-zA-Z]", " ", text)  # Remove punctuation
    text = re.sub(" +", " ", text)  # Remove extra spaces
    text = re.sub("\s+", " ", text)
    text = re.sub(url_regex, "", text)  # Remove every url
    # text = re.sub('\w*\d\w*', '', text) # remove whole word if starts with number
    # text = re.sub(r'(\s)#\w+', '', text) # remove whole word if starts with #
    text = text.strip()  # strip text
    text = text.lower()  # convert all text to lowercase
    return text


alay_dict_map = dict(zip(alay_dict["original"], alay_dict["replacement"]))


def normalization(text):
    return " ".join(
        [
            alay_dict_map[word] if word in alay_dict_map else word
            for word in text.split(" ")
        ]
    )


# Function 4b
# -------------
preprocess = lambda x: preprocessing(x)


def labelling(result):
    if result == 1:
        return "HateSpeechWeak"
    elif result == 2:
        return "HateSpeechModerate"
    elif result == 3:
        return "HateSpeechStrong"
    else:
        return "NonHateSpeech"


def multi_hatespeech_detection(df, data_column):
    label_hs = []
    prob = []
    for index, row in df.iterrows():

        encoded_inputs = tokenizer.encode_plus(
            str(row[data_column]), add_special_tokens=True
        )
        subwords = encoded_inputs["input_ids"]

        subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

        logits = model(subwords)[0]
        label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()
        probability = f"{F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}%"

        label_hs.append(labelling(label))
        prob.append(probability)

    df[f"Label"] = label_hs
    df[f"Probability"] = prob
    return df


def single_hatespeech_detection(text):
    encoded_inputs = tokenizer.encode_plus(text, add_special_tokens=True)
    subwords = encoded_inputs["input_ids"]

    subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

    logits = model(subwords)[0]
    label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()
    probability = f"{F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}"

    hasil = labelling(label)

    return hasil, probability
