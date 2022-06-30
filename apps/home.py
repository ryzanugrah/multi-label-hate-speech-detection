"""Import the libraries."""

import streamlit as st
from PIL import Image


def main():
    """Main app."""

    st.title("Deteksi Hate Speech Bahasa Indonesia")
    st.caption(
        "Sistem Deteksi Hate Speech Berbahasa Indonesia Menggunakan Metode "
        "Indonesian Bidirectional Encoder Representations from Transformers (IndoBERT)"
    )

    with st.container():
        st.write(
            "Sistem deteksi hate speech bahasa Indonesia adalah website yang digunakan untuk "
            "mendeteksi apakah kalimat berbahasa Indonesia yang ingin diklasifikasikan merupakan "
            "hate speech atau non-hate speech. Sistem ini akan berfokus pada teks atau kalimat berbahasa Indonesia."
        )

    st.markdown("""---""")

    st.header("Tentang Website")

    with st.expander("Latar belakang pembuatan website"):
        with st.container():
            st.subheader("Abstrak")

            st.write(
                "Saat ini penggunaan internet paling banyak digunakan untuk mengakses media sosial. "
                "Namun banyak masyarakat yang menyalahgunakan media sosial dengan memposting kalimat "
                "bernada ujaran kebencian atau hate speech. Dengan merebaknya postingan media sosial di "
                "Indonesia yang bernada ujaran kebencian atau hate speech, membuat suatu sistem yang dapat "
                "mendeteksi hate speech pada teks bahasa Indonesia merupakan langkah yang sangat tepat agar "
                "masyarakat tidak lagi sembarangan dalam memposting kalimat bernada hate speech.\n\nUntuk itu "
                "penulis membuat sistem deteksi hate speech berbasis Deep Learning pada teks berbahasa Indonesia "
                "dengan metode IndoBERT yang digunakan untuk mengklasifikasikan teks yang bermakna hate speech "
                "dan non-hate speech. Data pada penelitian ini didapatkan dari dataset yang sudah dibangun dan "
                "dilabeli dari penelitian (Ibrohim & Budi, 2019). Total data yang didapatkan sebanyak 13169 data "
                "yang terdiri dari 4 label, yaitu Non-Hate Speech sebanyak 5860 data, Weak Hate Speech sebanyak "
                "3383 data, Moderate Hate Speech sebanyak 1705 data, dan Strong Hate Speech sebanyak 2221 data.\n\nHasil "
                "pada penelitian ini menghasilkan nilai f1-score (macro avg) sebesar 76% dengan set hyperparameter yang digunakan "
                "yaitu batch size 16, learning rate 2e-5, dan epoch 3."
            )

        st.markdown("""---""")

        with st.container():
            st.subheader("Tujuan")

            st.caption("Tujuan dari pembuatan website ini adalah:")

            st.write(
                "1. Dapat mengklasifikasikan teks berbahasa Indonesia menjadi 4 jenis "
                "klasifikasi yaitu weak hate speech, moderate hate speech, "
                "strong hate speech, dan non-hate speech.\n\n2.	Untuk mengimplementasikan metode IndoBERT "
                "dalam membuat suatu sistem yang dapat mendeteksi hate speech pada teks berbahasa Indonesia.\n\n"
                "3.	Untuk membuat sistem berbasis website yang dapat mendeteksi hate speech pada teks berbahasa Indonesia."
            )

        st.markdown("""---""")

        with st.container():
            st.subheader("Manfaat")

            st.caption("Manfaat dari pembuatan website ini adalah:")

            st.write(
                "1.	Menghasilkan sistem berbasis website yang dapat mendeteksi hate speech pada teks berbahasa Indonesia.\n\n"
                "2.	Mengurangi jumlah pengguna yang menyalahgunakan media sosial dengan memposting kalimat bernada ujaran kebencian atau hate speech.\n\n"
                "3.	Membantu masyarakat dan aparat hukum agar dapat lebih mudah untuk mengklasifikasikan postingan yang bermakna "
                "hate speech dan non-hate speech di media sosial."
            )

    with st.expander("Tentang pengembangan sistem"):
        with st.container():
            st.subheader("Pengembangan Sistem")

            st.write(
                "Pengembangan sistem berbasis website ini memanfaatkan framework Streamlit yang digunakan "
                "untuk memudahkan pengembang dalam membangun aplikasi berbasis website dengan menggunakan "
                "bahasa pemrograman Python. Website yang telah dikembangkan ini memiliki beberapa halaman, "
                "yaitu halaman awal, halaman training model, halaman hasil terbaik, dan halaman deteksi."
            )

    with st.expander("Penjelasan singkat halaman website"):
        with st.container():
            st.subheader("Halaman Awal")

            st.write(
                "Halaman awal merupakan halaman paling awal yang akan dilihat oleh pengguna ketika pertama kali "
                "mengakses website ini. Halaman ini menampilkan judul halaman, informasi mengenai website, cara "
                "penggunaan website, dan penjelasan mengenai semua fitur yang ada pada website."
            )

        st.markdown("""---""")

        with st.container():
            st.subheader("Halaman Training")

            st.write(
                "Halaman training menampilkan form parameter pengujian yang tersedia untuk dikonfigurasi "
                "nilainya oleh pengguna, dan terdapat tombol untuk memulai training model. Pada halaman ini "
                "pengguna dapat melakukan pengaturan pada parameter, lalu pengguna dapat langsung melakukan "
                "simulasi training model dengan lama waktu eksekusi program dan hasil klasifikasi tergantung "
                "parameter yang sebelumnya sudah diatur oleh pengguna. Apabila proses training model sudah selesai, "
                "sistem akan menampilkan hasil training model yang telah dilakukan, seperti waktu lamanya eksekusi, "
                "diagram hasil pelatihan model, nilai performance metrics, dan confusion matrix."
            )

        st.markdown("""---""")

        with st.container():
            st.subheader("Halaman Hasil Pelatihan Terbaik")

            st.write(
                "Halaman hasil terbaik menampilkan hasil pelatihan dan pengujian dari model trained terbaik. "
                "Pada halaman ini pengguna dapat melihat hasil pelatihan dan pengujian model berupa nilai dari "
                "hasil pelatihan setiap epoch-nya, diagram batang yang menampilkan hasil training dan loss "
                "setiap epoch-nya, hasil performance metrics, tabel classification report, dan confusion matrix."
            )

        st.markdown("""---""")

        with st.container():
            st.subheader("Halaman Deteksi Hate Speech Single Tweet")

            st.write(
                "Halaman deteksi hate speech single tweet menampilkan form deteksi yang bisa diisi kata atau "
                "kalimat berbahasa Indonesia yang ingin dideteksi. Pada halaman ini pengguna dapat melakukan "
                "deteksi hate speech pada teks atau kalimat berbahasa Indonesia. Setelah proses klasifikasi "
                "selesai, halaman akan menampilkan hasil klasifikasi apakah kata yang dimasukkan termasuk "
                "ke dalam kelas hate speech atau non-hate speech."
            )

        st.markdown("""---""")

        with st.container():
            st.subheader("Halaman Deteksi Hate Speech Multi Tweet")

            st.write(
                "Halaman deteksi hate speech multi tweet menampilkan form deteksi yang bisa diisi kata atau "
                "hashtag berbahasa Indonesia yang ingin dideteksi, beserta jumlah tweet yang ingin diklasifikasi. "
                "Pada halaman ini pengguna dapat melakukan deteksi hate speech pada tweet berbahasa Indonesia "
                "berdasarkan kata atau hashtag yang dimasukkan. Setelah proses klasifikasi selesai, halaman akan "
                "menampilkan tweet yang diambil dari Twitter secara langsung lalu menampilkan hasil klasifikasi "
                "apakah tweet tersebut termasuk ke dalam kelas hate speech atau non-hate speech dan output-nya "
                "dapat dilihat dalam bentuk tabel dataframe dan dapat disimpan oleh pengguna ke dalam format CSV. "
            )

    with st.expander("Tentang pengembang"):
        with st.container():
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("Penulis")

                image = Image.open("./data/input/image/140810180049.jpg")

                st.image(image)
                st.write("Rizky Anugerah")
                st.write("NPM 140810180049")

            with col2:
                st.write("Pembimbing Utama")

                image = Image.open("./data/input/image/198507042015042003.jpg")

                st.image(image)
                st.write("Dr. Intan Nurma Yulita, MT")
                st.write("NIP 198507042015042003")

            with col3:
                st.write("Co. Pembimbing")

                image = Image.open("./data/input/image/198412112015041002.jpg")

                st.image(image)
                st.write("Aditya Pradana, S. T., M. Eng.")
                st.write("NIP 198412112015041002")


"""
Using the special variable
__name__
to run main function
"""

if __name__ == "__main__":
    main()
