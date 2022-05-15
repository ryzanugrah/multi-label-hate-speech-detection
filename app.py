import streamlit as st
from multiapp import MultiApp
from apps import singletweet, multitweet

app = MultiApp()

app.add_app("Single Tweet", singletweet.app)
app.add_app("Multi Tweet", multitweet.app)

app.run()