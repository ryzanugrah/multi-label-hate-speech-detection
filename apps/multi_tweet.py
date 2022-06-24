"""Import the libraries."""

import time

import streamlit as st
from st_material_table import st_material_table

from apps import utils as ut


def main():
    """Main app."""

    st.title("Deteksi Hate Speech Multi Tweet")
    st.caption(
        "Sistem deteksi hate speech bahasa Indonesia dengan model "
        "Indonesian Bidirectional Encoder Representations from Transformers (IndoBERT)"
    )

    st.info("Masukkan kata kunci atau hashtag Twitter di bawah.")

    # Type of search (by keyword or hashtag)
    type_of_search = st.selectbox(
        "Tipe Pencarian",
        ("Kata Kunci", "Hashtag"),
        key="type",
        help="Pilih kata kunci atau hashtag.",
    )

    # Text input
    text_input = st.text_input(
        "Kata kunci atau hashtag",
        key="multi_tweet",
        help="Pastikan kata kunci atau hashtag tidak mengandung spasi.",
        placeholder="Masukkan kata kunci atau hashtag",
    )

    # Number of tweets
    num_of_tweets = st.number_input(
        "Jumlah Tweet Maksimal",
        key="num_of_tweets",
        min_value=10,
        max_value=150,
        value=10,
        step=10,
        help=("Masukkan jumlah maksimal tweet yang ingin dicari (min=10, max=150)."),
    )

    # Start program
    if st.button("Mulai Deteksi", help="Klik untuk memulai deteksi."):
        if not text_input:
            st.error(
                "Tidak ada kata kunci atau hashtag yang dimasukkan. Masukkan kata kunci atau hashtag terlebih dahulu."
            )
            st.stop()

        container = st.container()

        with container:
            """Start detection"""

            st.markdown("""---""")

            st.header("Hasil Deteksi Hate Speech")

            with st.expander("Hasil pencarian tweet", expanded=True):
                st.subheader("Data Twitter")

                with st.spinner("Mengambil data dari Twitter..."):
                    # Get twitter data using Tweepy
                    df_tweets, df_new_tweets = ut.twitter_api(
                        type_of_search, text_input, num_of_tweets
                    )

                    # clean tweets with text preprocessing
                    df_tweets["clean_text"] = df_tweets.full_text.apply(
                        ut.preprocess_input
                    )
                    user_num_tweets = str(num_of_tweets)
                    total_tweets = len(df_tweets["full_text"])

                    if type_of_search == "Kata Kunci":
                        st.success(
                            "Sistem telah mencari "
                            + user_num_tweets
                            + " data tweet dengan kata kunci "
                            + f'"{text_input}"'
                        )

                        with st.sidebar:
                            st.header("Detail Deteksi")
                            st.caption(
                                "Kata kunci yang dicari akan ditampilkan di bawah."
                            )
                            st.info(
                                "Kata kunci: "
                                + f"{text_input}"
                                + "\n\nJumlah maks tweet: "
                                + f"{user_num_tweets}"
                            )

                    else:
                        st.success(
                            "Sistem telah mencari "
                            + user_num_tweets
                            + " data tweet dengan #"
                            + f"{text_input}"
                        )

                        with st.sidebar:
                            st.header("Detail Deteksi")
                            st.caption("Hashtag yang dicari akan ditampilkan di bawah.")
                            st.info(
                                "Hashtag: "
                                + f"{text_input}"
                                + "\n\nJumlah maks tweet: "
                                + f"{user_num_tweets}"
                            )

            with st.expander("Identifikasi hate speech multi tweet", expanded=True):
                st.subheader("Deteksi Hate Speech")

                with st.spinner("Mengidentifikasi tweet..."):
                    ut.multi_hatespeech_detection(df_tweets, "clean_text")
                    # Select columns to output
                    df_hs = df_tweets[["created_dttime", "full_text", "Label"]]
                    df_hs = df_hs.rename(
                        columns={"created_dttime": "Waktu", "full_text": "Tweet"}
                    )
                    hs_group = (
                        df_hs.groupby("Label").agg({"Label": "count"}).transpose()
                    )

                    df_new_tweets = df_tweets[
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
                    df_new_tweets = df_new_tweets.rename(
                        columns={
                            "created_dttime": "Time",
                            "user": "Username",
                            "full_text": "Tweet",
                            "clean_text": "Tweet Cleaned",
                        }
                    )

                    time.sleep(2)

                    st.success("Deteksi Hate Speech berhasil dilakukan.")

            with st.expander("Hasil deteksi yang didapatkan yaitu:", expanded=True):
                result_non_hs = round(
                    (max(hs_group["Non Hate Speech"]) / total_tweets) * 100
                )
                results_hs = 100 - round(
                    ((max(hs_group["Non Hate Speech"]) / total_tweets) * 100)
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Tweet Non Hate Speech", value=f"{result_non_hs}%")

                with col2:
                    st.metric("Tweet Hate Speech", value=f"{results_hs}%")

            st.markdown("""---""")

            with st.container():
                st.subheader("Data Twitter")
                st.caption("Unduh tabel di bawah untuk melihat info lengkapnya.")
                st.markdown(
                    ut.get_table_download_df(
                        df_new_tweets, "multi_tweet.csv", "Unduh tabel"
                    ),
                    unsafe_allow_html=True,
                )
                st_material_table(df_hs)


"""
Using the special variable
__name__
to run main function
"""

if __name__ == "__main__":
    main()
