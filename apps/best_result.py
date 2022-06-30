"""Import the libraries."""

import pandas as pd
import streamlit as st
from PIL import Image


def main():
    """Main app."""

    with st.sidebar:
        st.header("Info Label")
        st.caption(
            "Keterangan label pada dataset akan ditampilkan di bawah."
        )
        st.info(
            "Non-Hate Speech: 0"
            "\n\nHate Speech Lemah: 1"
            "\n\nHate Speech Menengah: 2"
            "\n\nHate Speech Kuat: 3"
        )

    st.title("Hasil Pelatihan Model Terbaik")

    with st.container():
        st.write(
            "Hasil Pelatihan model IndoBERT terbaik didapatkan dari pelatihan model yang telah "
            "dilakukan. Pada proses pelatihan model, dilakukanlah fine-tuning agar model yang "
            "dihasilkan dapat melakukan tugas spesifik, contohnya pada kasus ini adalah mendeteksi "
            "kalimat hate speech bahasa Indonesia.\n\nAgar fine-tuning dapat menghasilkan model yang "
            "optimal, dilakukanlah hyperparameter tuning dengan mengatur nilai-nilai hyperparameter "
            "dengan beberapa kombinasi hyperparameter tuning yang telah diuji sebagai berikut:"
        )
        st.info(
            "\n\nBatch Size: 16 dan 32\n\nLearning Rate: 2e-5, 3e-5, dan 5e-5\n\nEpoch: 2, 3, dan 4"
        )

    st.markdown("""---""")

    st.header("Hasil Terbaik")

    with st.expander("Konfigurasi Hyperparameter yang di atur", expanded=True):
        st.subheader("Konfigurasi Hyperparameter")
        st.write(
            "Hasil yang ditampilkan di bawah ini didapatkan dari pelatihan model "
            "dengan konfigurasi hyperparameter sebagai berikut:"
        )
        st.info("\n\nBatch Size: 16\n\nLearning Rate: 2e-5\n\nEpoch: 3")

    with st.expander("Diagram hasil pelatihan", expanded=True):
        st.subheader("Diagram Hasil Pelatihan")

        col1, col2 = st.columns(2)

        with col1:
            with st.container():
                image = Image.open("./data/output/best/accuracy.png")
                st.image(image, caption="Accuracy")

            with st.container():
                image = Image.open("./data/output/best/recall.png")
                st.image(image, caption="Recall")

        with col2:
            with st.container():
                image = Image.open("./data/output/best/precision.png")
                st.image(image, caption="Precision")

            with st.container():
                image = Image.open("./data/output/best/f1_score.png")
                st.image(image, caption="F1-score")

        with st.container():
            image = Image.open("./data/output/best/loss.png")
            st.image(image, caption="Loss")

    with st.expander("Tabel hasil pelatihan", expanded=True):
        st.subheader("Tabel Hasil Pelatihan")

        df = pd.read_csv("./data/output/best/training_result.csv", index_col=0)

        st.dataframe(df)

    with st.expander("Classification Report", expanded=True):
        st.subheader("Tabel Classification Report")

        df = pd.read_csv("./data/output/best/classification_report.csv", index_col=0)

        st.dataframe(df)

    with st.expander("Confusion Matrix", expanded=True):
        with st.container():
            st.subheader("Confusion Matrix")

            image = Image.open("./data/output/best/confusion_matrix.png")
            st.image(image, caption="Confusion Matrix")

    with st.expander("Hasil Klasifikasi", expanded=True):
        with st.container():
            st.subheader("Matriks Performa")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Accuracy", value="79%")
                st.metric("F1-score (Micro Avg)", value="79%")

            with col2:
                st.metric("Precision", value="75%")
                st.metric("F1-score (Macro Avg)", value="76%")

            with col3:
                st.metric("Recall", value="77%")
                st.metric("F1-score (Weighted Avg)", value="80%")


"""
Using the special variable
__name__
to run main function
"""

if __name__ == "__main__":
    main()
