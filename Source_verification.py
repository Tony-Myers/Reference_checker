import streamlit as st
import requests
import json
import os
import re
from typing import Dict, List, Tuple, Optional
import time
import openai
from semanticscholar import SemanticScholar
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_api_keys():
    """Load API keys from Streamlit secrets, environment variables, or .env file."""
    try:
        openai_api_key = st.secrets["openai"]["api_key"]
        semanticscholar_api_key = st.secrets.get("semanticscholar", {}).get("api_key")
        app_password = st.secrets.get("app_password")
        logger.info("API keys loaded from Streamlit secrets")
    except (KeyError, FileNotFoundError):
        logger.info("API keys not found in Streamlit secrets, trying environment variables")
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            logger.info("python-dotenv not installed, using environment variables directly")
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        semanticscholar_api_key = os.environ.get("SEMANTICSCHOLAR_API_KEY", None)
        app_password = os.environ.get("APP_PASSWORD")

    if not openai_api_key:
        logger.error("OpenAI API key not found in secrets or environment variables")
        raise ValueError("Missing API key for OpenAI")
    return openai_api_key, semanticscholar_api_key, app_password


class ReferenceVerifier:
    """Class for verifying academic references and claims."""

    def __init__(self, semantic_scholar_client=None):
        self.openai_client = openai.OpenAI(api_key=openai.api_key)
        self.ss = semantic_scholar_client

    # --- All other methods remain unchanged and correctly indented ---
    # For brevity I omit them here — they should be exactly as in your original script.

    def verify_claim_with_reference(self, claim: str, reference_details: Dict) -> Tuple[bool, str]:
        """Verify if a claim is supported by the reference."""
        abstract = reference_details.get('abstract', '')
        paper_info = (
            f"Title: {reference_details.get('title', '')}\n"
            f"Authors: {', '.join(reference_details.get('authors', []))}\n"
            f"Year: {reference_details.get('year', '')}\n"
            f"Journal: {reference_details.get('journal', '')}\n"
            f"Abstract: {abstract}"
        )

        prompt = f"""
Evaluate whether the following claim is supported by the academic reference provided.

Claim:
"{claim}"

Reference information:
{paper_info}

Format your response as a JSON object with:
- "is_supported": boolean
- "confidence": number between 0-1
- "reasoning": explanation
- "evidence": text from the abstract
"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            return result.get('is_supported', False), result.get('reasoning', '')
        except Exception as e:
            logger.error(f"Error verifying claim: {str(e)}")
            return False, f"Error during verification: {str(e)}"


def create_streamlit_app():
    """Create and run the Streamlit app for the reference verifier."""
    st.set_page_config(page_title="Academic Reference Verifier", page_icon="📚", layout="wide")
    st.title("Academic Reference Verification Agent")

    try:
        openai_api_key, semanticscholar_api_key, app_password = load_api_keys()
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        st.stop()

    authenticated = False
    if app_password:
        st.sidebar.title("Authentication")
        if st.sidebar.text_input("Enter password", type="password") == app_password:
            authenticated = True
            st.sidebar.success("Authentication successful! ✅")
        else:
            st.sidebar.error("Incorrect password ❌")
    else:
        authenticated = True

    if not authenticated:
        st.info("Please enter the password in the sidebar to access this application.")
        st.stop()

    openai.api_key = openai_api_key
    semantic_scholar_client = None
    if semanticscholar_api_key:
        try:
            semantic_scholar_client = SemanticScholar(api_key=semanticscholar_api_key)
        except Exception as e:
            st.sidebar.warning("Semantic Scholar API not available")

    verifier = ReferenceVerifier(semantic_scholar_client=semantic_scholar_client)

    with st.form("reference_form"):
        ai_output = st.text_area("Paste AI-generated text with academic references", height=300)
        submitted = st.form_submit_button("Verify References")

    if submitted and ai_output:
        references = verifier.extract_references(ai_output)
        claims = verifier.extract_claims_with_citations(ai_output)

        st.subheader("Extracted References")
        for i, ref in enumerate(references):
            st.write(f"{i+1}. {ref.get('title')} ({ref.get('year')})")

        st.subheader("Extracted Claims")
        for i, claim_obj in enumerate(claims):
            st.write(f"{i+1}. {claim_obj.get('claim')} [Citation: {claim_obj.get('citation')}]")

        st.subheader("Reference Verification Results")
        for i, ref in enumerate(references):
            with st.expander(f"Reference {i+1}: {ref.get('title')}"):
                exists, details, source = verifier.verify_reference_exists(ref)
                if exists:
                    st.success(f"✅ Reference found in {source}")
                    st.json(details)
                else:
                    st.error("❌ Reference not found")

if __name__ == "__main__":
    create_streamlit_app()
