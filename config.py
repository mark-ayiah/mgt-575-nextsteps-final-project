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
    if not api_key and hasattr(st, "secrets") and "openai" in st.secrets:
        api_key = st.secrets["openai"]["api_key"]
    
    # If still not found, ask the user
    if not api_key:
        with st.sidebar.expander("🔑 OpenAI API Key", expanded=True):
            api_key = st.text_input(
                "Enter your OpenAI API Key:",
                type="password",
                help="Get your API key from https://platform.openai.com/account/api-keys",
                placeholder="sk-...",
                key="openai_api_key_input"
            )
            st.info(
                """
                Your API key is stored only in your local session and not saved by the application.
                You'll need to re-enter it if you refresh or restart the app.
                """
            )
            
            # Add a verification indicator
            if api_key:
                if len(api_key) > 20 and api_key.startswith("sk-"):
                    st.success("✅ API key format looks valid")
                else:
                    st.error("❌ API key format looks invalid. It should start with 'sk-' and be longer than 20 characters.")
    
    return api_key