# 1.1: Load libraries
# ------------------------------------#
from numpy import e
import streamlit as st
from st_material_table import st_material_table
from streamlit_metrics import metric, metric_row
import pandas as pd
import tweepy as tw
import time
import function as tf


def app():
    st.title("Deteksi Hate Speech Bahasa Indonesia Multi Tweet")

    st.markdown(
        """
    Cari hashtag atau keyword Twitter di sidebar untuk menjalankan deteksi hate speech!
    """
    )
    # 2.2: Sidebar Setup
    ## 2.1.1: Sidebar Title
    st.sidebar.header("Pilih Masukkan Pencarian")  # sidebar title

    ## 2.2.2: Sidebar Input Fields
    with st.form(key="form_1"):
        with st.sidebar:
            user_word_entry = st.text_input(
                "1. Masukkan satu hashtag atau keyword",
                help="Pastikan hashtag atau kata kunci tidak mengandung spasi",
            )
            select_hashtag_keyword = st.radio(
                "2. Cari hashtags atau keywords?",
                ("Hashtag", "Keyword"),
                help="Hanya mencari hashtag akan memberikan hasil yang lebih sedikit",
            )
            num_of_tweets = st.number_input(
                "3. Jumlah maksimal tweet",
                min_value=50,
                max_value=1000,
                value=50,
                step=50,
                help="Mengembalikkan tweet terbaru dalam 7 hari terakhir",
            )
            st.sidebar.text("")  # spacing
            submitted1 = st.form_submit_button(label="Jalankan Deteksi Hate Speech 🚀")

    if submitted1:
        if not user_word_entry:
            st.warning("Silakan masukkan hashtag atau keyword!")
            st.stop()

        # Run function 2: Get twitter data
        df_tweets, df_new = tf.twitter_get(
            select_hashtag_keyword, user_word_entry, num_of_tweets
        )

        # Run function #4: Round 1 text cleaning (convert to lower, remove numbers, @, punctuation, numbers. etc.)
        df_tweets["clean_text"] = df_tweets.full_text.apply(tf.preprocess)
        df_tweets["clean_text"] = df_tweets["clean_text"].apply(tf.normalization)
        user_num_tweets = str(num_of_tweets)
        total_tweets = len(df_tweets["full_text"])

        # 4.1: UX Messaging
        # Loading message for users
        with st.spinner("Mengambil data dari Twitter..."):
            time.sleep(5)
            # Keyword or hashtag
            if select_hashtag_keyword == "Hashtag":
                st.success(
                    "🎈Done! Anda telah mencari "
                    + user_num_tweets
                    + " tweets terakhir yang menggunakan #"
                    + user_word_entry
                )

            else:
                st.success(
                    "🎈Done! Anda telah mencari "
                    + user_num_tweets
                    + " tweets yang menggunakan kata kunci "
                    + user_word_entry
                )

        # 4.2: Hate Speech Detection
        st.header("Deteksi Hate Speech")

        tf.multi_hatespeech_detection(df_tweets, "clean_text")
        # Select columns to output
        df_hs = df_tweets[["created_dttime", "full_text", "Label"]]
        df_hs = df_hs.rename(
            columns={"created_dttime": "Datetime", "full_text": "Tweets"}
        )
        hs_group = df_hs.groupby("Label").agg({"Label": "count"}).transpose()

        df_new = df_tweets[
            [
                "created_dttime",
                "id",
                "user",
                "full_text",
                "clean_text",
                "Label",
                "Probability",
            ]
        ]
        df_new = df_new.rename(
            columns={
                "created_dttime": "Date & Time",
                "user": "Username",
                "full_text": "Tweet",
                "clean_text": "Clean Tweet",
            }
        )

        ## 4.2.1: Summary Card Metrics
        st.subheader("Summary")
        metric_row(
            {
                "Non Hate Speech Tweets": "{:.0%}".format(
                    max(hs_group.NonHateSpeech) / total_tweets
                ),
            }
        )
        st_material_table(df_hs)
        st.markdown("**Untuk lebih lengkap dapat mengunduh file csv dibawah ini**")
        st.markdown(tf.get_table_download_link(df_new), unsafe_allow_html=True)
