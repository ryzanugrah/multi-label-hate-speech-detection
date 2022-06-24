"""Import the libraries."""

import time
import timeit

import pandas as pd
import streamlit as st
import tensorflow as tf
import torch

from apps import utils as ut


def main():
    """Main app."""

    st.title("Pelatihan Model IndoBERT Hate Speech Bahasa Indonesia")

    # Check Tensorflow and Pytorch versions
    st.caption(
        "Pelatihan dengan TensorFlow v"
        + tf.__version__
        + " "
        + "dan PyTorch v"
        + torch.__version__
    )

    with st.container():
        st.write(
            "Pelatihan model adalah proses untuk membuat model baru berdasarkan model IndoBERT yang sudah "
            "dilatih sebelumnya, yang nantinya model yang dihasilkan pada proses pelatihan ini akan "
            "digunakan untuk memprediksi kalimat hate speech berbahasa Indonesia."
        )

    st.markdown("""---""")

    """
    In order for torch to use the GPU, we need to identify and specify
    the GPU as the device. Later, in our training loop,
    we will load data onto the device.
    """

    with st.container():
        st.header("Info perangkat")

        # If there's a GPU available
        if torch.cuda.is_available():
            # Tell PyTorch to use the GPU.
            device = torch.device("cuda")

            # Get the GPU device count
            device_count = str(len(tf.config.list_physical_devices("GPU")))

            # Get the GPU device name
            device_name = tf.test.gpu_device_name()

            st.info(
                "Jumlah GPU tersedia: "
                + device_count
                + "\n\nGPU: "
                + torch.cuda.get_device_name(0)
            )

        # If not
        else:
            st.error(
                "GPU tidak ditemukan. Masalah ini terjadi karena perangkat tidak memiliki GPU "
                "atau Python tidak dikonfigurasi untuk menggunakan GPU untuk melakukan pelatihan."
            )
            device = torch.device("cpu")

    ut.set_hyperparameter()

    st.write("")

    # Advanced settings
    with st.expander("Pengaturan lanjutan", expanded=True):
        global check_data_cleaning, check_case_folding, check_normalization, check_filtering, check_stemming, check_smote

        st.error(
            "Biarkan semua pengaturan di bawah jika tidak yakin dengan apa yang diubah."
        )

        with st.container():
            st.subheader("Text Preprocessing")

            check_data_cleaning = st.checkbox(
                "Lakukan Data Cleaning",
                value=True,
                help="Menghapus beberapa karakter yang tidak dibutuhkan menggunakan regular expression.",
            )
            check_case_folding = st.checkbox(
                "Lakukan Case Folding",
                value=True,
                help="Mengubah semua huruf menjadi huruf kecil.",
            )
            check_normalization = st.checkbox(
                "Lakukan Normalization",
                value=True,
                help="Merubah kata tidak baku menjadi baku.",
            )
            check_filtering = st.checkbox(
                "Lakukan Filtering",
                value=True,
                help="Menghapus kata-kata yang tidak memiliki makna.",
            )
            check_stemming = st.checkbox(
                "Lakukan Stemming",
                value=True,
                help="Mengubah kata berimbuhan menjadi bentuk dasarnya.",
            )

        with st.container():
            st.subheader("SMOTE")

            check_smote = st.checkbox(
                "Lakukan Oversampling Data",
                value=True,
                help="Menambah jumlah data kelas minor agar sama dengan jumlah data kelas mayor.",
            )

    st.write("")
    st.warning(
        "Waktu dan hasil pelatihan model yang didapat bisa berbeda-beda "
        "pada setiap perangkat. Atur nilai hyperparameter sesuai dengan "
        "spesifikasi perangkat yang digunakan agar mendapatkan hasil yang optimal."
    )

    # Start program
    if st.button("Mulai pelatihan", help="Klik untuk memulai pelatihan model."):
        container = st.container()

        with container:
            """Start training model"""

            start = timeit.default_timer()

            st.markdown("""---""")

            st.header("Hasil Pelatihan Model")

            ut.set_seed(0)

            with st.expander("Impor data-data yang diperlukan", expanded=True):
                with st.container():
                    st.subheader("Dataset Hate Speech Multi Label")
                    st.caption("Dataset dari penelitian (Ibrohim & Budi, 2019)")

                    with st.spinner("Mengimpor dataset..."):
                        # Forked from 'https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/re_dataset.csv (Ibrohim & Budi, 2019).
                        df = "https://raw.githubusercontent.com/ryzanugrah/id-multi-label-hate-speech-and-abusive-language-detection/master/re_multi_label_dataset.csv"
                        df = pd.read_csv(df, encoding="utf-8")
                        df.rename({"Tweet": "text"}, axis=1, inplace=True)

                        time.sleep(2)

                    # Show the first 5 dataset rows
                    st.write(df)
                    st.write("Jumlah Data: ", len(df))

                st.markdown("""---""")

                with st.container():
                    st.subheader("Kamus Kosakata Baku")
                    st.caption(
                        "Kamus kosakata baku dari penelitian (Ibrohim & Budi, 2019)"
                    )

                    with st.spinner("Mengimpor kosakata baku..."):
                        # Import Slang Dictionary from (Ibrohim & Budi, 2019)
                        alay_dict = "https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/new_kamusalay.csv"
                        alay_dict = pd.read_csv(
                            alay_dict, encoding="latin-1", header=None
                        )
                        alay_dict = alay_dict.rename(
                            columns={0: "original", 1: "replacement"}
                        )

                        time.sleep(2)

                    # Show Slang Dictionary
                    st.dataframe(alay_dict)
                    st.write("Jumlah Data: ", len(alay_dict))

                st.markdown("""---""")

                with st.container():
                    st.subheader("Stopword")
                    st.caption("Stopword dari penelitian (Tala, F. Z., 2003)")

                    with st.spinner("Mengimpor stopword..."):
                        # Initiate (Tala, F. Z., 2003) stopword
                        tala_stopword_dict = pd.read_csv(
                            "https://raw.githubusercontent.com/ryzanugrah/stopwords-id/master/stopwords-id.txt",
                            header=None,
                        )
                        tala_stopword_dict = tala_stopword_dict.rename(
                            columns={0: "stopword"}
                        )

                        time.sleep(2)

                    # Show stopwords
                    st.dataframe(tala_stopword_dict)
                    st.write("Jumlah Data: ", len(tala_stopword_dict))

                st.success("Data berhasil diimpor.")

            with st.expander("Bagan dataset", expanded=True):
                """Show Data on Chart."""

                with st.spinner("Menampilkan Bagan..."):
                    time.sleep(2)

                col1, col2 = st.columns(2)

                with col1:
                    # Histogram
                    st.subheader("Diagram Batang")
                    ut.show_histogram(df)

                with col2:
                    # Donut chart
                    st.subheader("Diagram Donat")

                    # Count value on each label
                    non_hs_label = df.loc[df.HS == 0, "HS"].count()
                    hs_label = df.loc[df.HS == 1, "HS"].count()
                    hs_weak_label = df.loc[df.HS_Weak == 1, "HS_Weak"].count()
                    hs_moderate_label = df.loc[
                        df.HS_Moderate == 1, "HS_Moderate"
                    ].count()
                    hs_strong_label = df.loc[df.HS_Strong == 1, "HS_Strong"].count()

                    inner_label = ["Non_HS", "HS_Weak", "HS_Moderate", "HS_Strong"]
                    outer_label = ["Non_HS", "HS"]
                    count_inner_data = [
                        non_hs_label,
                        hs_weak_label,
                        hs_moderate_label,
                        hs_strong_label,
                    ]
                    count_outer_data = [non_hs_label, hs_label]

                    ut.show_pie(
                        inner_label,
                        outer_label,
                        count_inner_data,
                        count_outer_data,
                        "Label",
                    )

            ut.text_preprocessing(df)
            ut.index_classification(df)

            with st.expander(
                "Menghapus nilai data yang hilang pada dataset", expanded=True
            ):
                """Remove Missing Values."""

                with st.spinner("Menghapus missing values..."):
                    time.sleep(2)

                    df_indexed = pd.read_csv(
                        "./data/output/dataset/dataset_preprocessed.csv"
                    )
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Sebelum")
                        st.write(df_indexed.isnull().sum())
                        st.write("Jumlah Data: ", len(df))

                    with col2:
                        df_indexed = df_indexed[df_indexed["text"].notna()]
                        st.subheader("Setelah")
                        st.write(df_indexed.isnull().sum())
                        st.write("Jumlah Data: ", len(df_indexed))

                # Save dataset
                df_indexed.to_csv(
                    "./data/output/dataset/dataset_preprocessed.csv", index=False
                )

            ut.split_dataset()
            ut.fine_tuning()
            ut.evaluate_model()

            stop = timeit.default_timer()
            execution_time = int(round((stop - start)))

            ut.elapsed_time(execution_time)

            if st.button("Reset hasil", help="Klik untuk mereset hasil pelatihan."):
                container.empty()


"""
Using the special variable
__name__
to run main function
"""

if __name__ == "__main__":
    main()
