import os
import streamlit as st

def load_openai_api_key():
    """
    Load OpenAI API key from environment variable or Streamlit secrets.
    If not found, provide a text input in the sidebar for the user to enter it.
    
    Returns:
        str: The OpenAI API key
    """
    # First check environment variable
    api_key = os.environ.get("OPENAI_API_KEY")

    # Then check Streamlit secrets
    if not api_key and "openai" in st.secrets:
        api_key = st.secrets["openai"].get("api_key", "")

    # Then prompt the user
    if not api_key:
        with st.sidebar.expander("🔑 OpenAI API Key", expanded=True):
            api_key = st.text_input(
                "Enter your OpenAI API Key:",
                type="password",
                placeholder="sk-...",
                help="You can get your key from https://platform.openai.com/account/api-keys"
            )
            st.info("Your key is stored locally for this session only and not saved by the app.")

    # Set it into environment for global access
    if api_key and api_key.startswith("sk-"):
        os.environ["OPENAI_API_KEY"] = api_key

    return api_key
