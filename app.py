import streamlit as st

from apps import best_result, home, multi_tweet, single_tweet, training_model
from apps import utils as ut
from multiapp import MultiApp

# Configures the default settings of the page
st.set_page_config(
    page_title="Deteksi Hate Speech Bahasa Indonesia",
    page_icon=":loudspeaker:",
    menu_items={
        "Get Help": None,
        "Report a bug": "https://www.instagram.com/ryzanugrah/",
        "About": "Sistem Deteksi Hate Speech Berbahasa Indonesia Menggunakan Metode Indonesian Bidirectional Encoder Representations from Transformers (IndoBERT)",
    },
)

# Load css for styling purposes
ut.css("./styles/style.css")

# Create an instance of the app
app = MultiApp()

# Add all applications/pages
app.add_app("Halaman Awal", home.main)
app.add_app("Pelatihan Model", training_model.main)
app.add_app("Hasil Pelatihan Terbaik", best_result.main)
app.add_app("Deteksi Single Tweet", single_tweet.main)
app.add_app("Deteksi Multi Tweet", multi_tweet.main)

# Main app
app.run()
