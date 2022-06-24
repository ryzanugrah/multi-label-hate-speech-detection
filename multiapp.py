"""
This file is the framework for generating multiple Streamlit applications
through an object oriented framework.
"""

import streamlit as st


# Define the MultiApp class to manage the multiple apps
class MultiApp:
    """Framework for combining multiple streamlit applications."""

    def __init__(self) -> None:
        """Constructor class to generate a list which will store all our applications as an instance variable."""
        self.apps = []

    def add_app(self, title, func) -> None:
        """Class Method to Add pages to the project.
        Args:
            title ([str]): The title of page which we are adding to the list of apps.

            func: Python function to render this page in Streamlit.
        """
        self.apps.append({"title": title, "function": func})

    def run(self):
        with st.container():
            with st.sidebar:
                st.title("Deteksi Hate Speech Bahasa Indonesia")
                st.write("")

        with st.container():
            # Drodown to select the page to run
            app = st.sidebar.selectbox(
                "Menu Navigasi",
                self.apps,
                format_func=lambda app: app["title"],
                help="Pilih halaman pada menu di bawah.",
            )

        # Run the app function
        app["function"]()
