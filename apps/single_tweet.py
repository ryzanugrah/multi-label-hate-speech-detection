"""Import the libraries."""

import time

import streamlit as st

from apps import utils as ut


def main():
    """Main app."""

    st.title("Deteksi Hate Speech Single Tweet")
    st.caption(
        "Sistem deteksi hate speech bahasa Indonesia dengan model "
        "Indonesian Bidirectional Encoder Representations from Transformers (IndoBERT)"
    )

    # Input text
    text = []
    text_input = st.text_input(
        "Teks atau Kalimat",
        key="single_tweet",
        help="Pastikan teks yang dimasukkan berbahasa Indonesia.",
        placeholder="Masukkan teks atau kalimat bahasa Indonesia",
    )
    text.append(text_input)

    # Start program
    if st.button("Mulai Deteksi", help="Klik untuk memulai deteksi."):
        if text_input != "":
            """Start detection"""

            st.markdown("""---""")

            st.header("Hasil Deteksi Hate Speech")

            with st.expander(
                "Hasil deteksi yang didapatkan yaitu:",
                expanded=True,
            ):
                with st.spinner("Sedang mengklasifikasi..."):
                    sentence = ut.preprocess_input(text_input)
                    result, probability = ut.single_tweet_detection(sentence)

                    time.sleep(2)

                    st.metric(f"{result}", value=f"{probability}%")

        else:
            st.error("Tidak ada teks yang dimasukkan. Masukkan teks terlebih dahulu.")


"""
Using the special variable
__name__
to run main function
"""

if __name__ == "__main__":
    main()
