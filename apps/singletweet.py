import streamlit as st
import pickle
import time
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig


def app():
    import function as tf

    st.title("Deteksi Hate Speech Bahasa Indonesia Single Tweet")

    text = []
    input_text = st.text_input("Masukkan Teks/Tweet")

    text.append(input_text)

    if st.button("Deteksi"):
        with st.spinner("Sedang mengidentifikasi..."):
            if input_text != "":
                sentence = tf.preprocess(input_text)
                sentence = tf.normalization(sentence)
                hasil, probability = tf.single_hatespeech_detection(sentence)

                st.success(
                    f"Teks tersebut hasilnya: **{hasil}** dengan **{probability}%** probability"
                )

            else:
                st.warning(
                    "[INFO] Tidak ada teks.. Silakan masukkan teks terlebih dahulu"
                )
