import streamlit as st
import requests
import json
import os
import re
from typing import Dict, List, Tuple, Optional
import time
import openai
from semanticscholar import SemanticScholar
import scholarly
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API keys from Streamlit secrets or environment variables
def load_api_keys():
    """Load API keys from Streamlit secrets, environment variables, or .env file."""
    # First try to get from Streamlit secrets
    try:
        openai_api_key = st.secrets["openai"]["api_key"]
        semanticscholar_api_key = st.secrets.get("semanticscholar", {}).get("api_key")
        app_password = st.secrets.get("app_password")
        
        logger.info("API keys loaded from Streamlit secrets")
        
    except (KeyError, FileNotFoundError):
        logger.info("API keys not found in Streamlit secrets, trying environment variables")
        
        # Fall back to environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            logger.info("python-dotenv not installed, using environment variables directly")
        
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        semanticscholar_api_key = os.environ.get("SEMANTICSCHOLAR_API_KEY", None)  # Optional
        app_password = os.environ.get("APP_PASSWORD")
    
    if not openai_api_key:
        logger.error("OpenAI API key not found in secrets or environment variables")
        raise ValueError("Missing API key for OpenAI")
    
    return openai_api_key, semanticscholar_api_key, app_password

# Initialize client - will be done after password validation in the Streamlit app
# API keys will be loaded when needed

class ReferenceVerifier:
    """Class for verifying academic references and claims."""
    
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=openai.api_key)
        self.ss = ss  # This might be None if the API key is unavailable
    
    def extract_references(self, ai_output: str) -> List[Dict]:
        """Extract academic references from an AI-generated output.
        
        Args:
            ai_output: Text containing academic references
            
        Returns:
            List of dictionaries with reference details
        """
        prompt = f"""
        Extract all academic references from the following text. For each reference, provide:
        1. Authors
        2. Title
        3. Year
        4. Journal/Conference/Publisher
        5. DOI (if present)
        6. URL (if present)
        
        Format the output as a JSON list of objects, with each object containing the fields above.
        
        Text:
        {ai_output}
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            if "references" in result:
                return result["references"]
            else:
                # Try to extract any JSON array from the response
                text = response.choices[0].message.content
                matches = re.findall(r'\[.*\]', text, re.DOTALL)
                if matches:
                    try:
                        return json.loads(matches[0])
                    except:
                        logger.error("Failed to parse references from extracted JSON array")
                return []
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from response: {response.choices[0].message.content}")
            return []
    
    def extract_claims_with_citations(self, ai_output: str) -> List[Dict]:
        """Extract claims and their associated citations from text.
        
        Args:
            ai_output: Text containing claims with citations
            
        Returns:
            List of dictionaries with claim text and citation markers
        """
        prompt = f"""
        Extract all claims with their associated citations from the following text.
        A claim is a factual statement that cites a source.
        
        For each claim, provide:
        1. The claim text
        2. The citation marker (e.g., "[1]", "(Smith et al., 2020)", etc.)
        
        Format the output as a JSON list of objects, with each object containing "claim" and "citation" fields.
        
        Text:
        {ai_output}
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            if "claims" in result:
                return result["claims"]
            else:
                # Handle case where the AI didn't use the expected structure
                text = response.choices[0].message.content
                matches = re.findall(r'\[.*\]', text, re.DOTALL)
                if matches:
                    try:
                        return json.loads(matches[0])
                    except:
                        logger.error("Failed to parse claims from extracted JSON array")
                return []
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from response: {response.choices[0].message.content}")
            return []
    
    def search_semanticscholar(self, reference: Dict) -> Optional[Dict]:
        """Search Semantic Scholar for a reference.
        
        Args:
            reference: Dictionary with reference details
            
        Returns:
            Dictionary with paper details if found, None otherwise
        """
        # Check if Semantic Scholar API is available
        if not self.ss:
            logger.warning("Semantic Scholar API not initialized, skipping this source")
            return None
            
        query = f"{reference.get('title', '')} {' '.join(reference.get('authors', []))}"
        try:
            papers = self.ss.search_paper(query, limit=5)
            
            # Check if we have results
            if not papers or len(papers) == 0:
                return None
            
            # Simple matching algorithm - could be improved
            best_match = None
            best_score = 0
            
            for paper in papers:
                score = 0
                # Match title (most important)
                if reference.get('title') and paper.title:
                    title_similarity = self._text_similarity(reference['title'].lower(), paper.title.lower())
                    score += title_similarity * 3  # Weight title heavily
                
                # Match authors
                if reference.get('authors') and paper.authors:
                    author_names = [author.name for author in paper.authors if author.name]
                    for ref_author in reference.get('authors', []):
                        if any(ref_author.lower() in auth.lower() for auth in author_names):
                            score += 1
                
                # Match year
                if reference.get('year') and paper.year and str(reference['year']) == str(paper.year):
                    score += 2
                
                if score > best_score:
                    best_score = score
                    best_match = paper
            
            # Require a minimum score to consider it a match
            if best_score > 3:
                return {
                    'title': best_match.title,
                    'authors': [author.name for author in best_match.authors if author.name],
                    'year': best_match.year,
                    'journal': best_match.journal,
                    'url': best_match.url,
                    'pdf_url': best_match.pdf_url,
                    'abstract': best_match.abstract
                }
            return None
        
        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {str(e)}")
            return None
    
    def search_google_scholar(self, reference: Dict) -> Optional[Dict]:
        """Search Google Scholar for a reference.
        
        Args:
            reference: Dictionary with reference details
            
        Returns:
            Dictionary with paper details if found, None otherwise
        """
        query = f"{reference.get('title', '')} {' '.join(reference.get('authors', []))}"
        try:
            search_query = scholarly.search_pubs(query)
            results = []
            
            # Get first 5 results
            for _ in range(5):
                try:
                    results.append(next(search_query))
                except StopIteration:
                    break
            
            if not results:
                return None
            
            # Simple matching algorithm
            best_match = None
            best_score = 0
            
            for result in results:
                score = 0
                # Match title
                if reference.get('title') and result.get('bib', {}).get('title'):
                    title_similarity = self._text_similarity(
                        reference['title'].lower(), 
                        result.get('bib', {}).get('title', '').lower()
                    )
                    score += title_similarity * 3
                
                # Match authors
                if reference.get('authors') and result.get('bib', {}).get('author'):
                    for ref_author in reference.get('authors', []):
                        if any(ref_author.lower() in auth.lower() for auth in result.get('bib', {}).get('author', [])):
                            score += 1
                
                # Match year
                if (reference.get('year') and result.get('bib', {}).get('pub_year') and 
                    str(reference['year']) == str(result.get('bib', {}).get('pub_year'))):
                    score += 2
                
                if score > best_score:
                    best_score = score
                    best_match = result
            
            # Require a minimum score to consider it a match
            if best_score > 3 and best_match:
                return {
                    'title': best_match.get('bib', {}).get('title'),
                    'authors': best_match.get('bib', {}).get('author'),
                    'year': best_match.get('bib', {}).get('pub_year'),
                    'journal': best_match.get('bib', {}).get('venue'),
                    'url': best_match.get('pub_url'),
                    'abstract': best_match.get('bib', {}).get('abstract')
                }
            return None
        
        except Exception as e:
            logger.error(f"Error searching Google Scholar: {str(e)}")
            return None
    
    def search_crossref(self, reference: Dict) -> Optional[Dict]:
        """Search CrossRef API for a reference.
        
        Args:
            reference: Dictionary with reference details
            
        Returns:
            Dictionary with paper details if found, None otherwise
        """
        title = reference.get('title', '')
        authors = reference.get('authors', [])
        year = reference.get('year', '')
        
        # Prepare query parameters
        params = {
            'query': title,
            'rows': 5  # Get up to 5 results
        }
        
        try:
            response = requests.get('https://api.crossref.org/works', params=params)
            response.raise_for_status()  # Raise exception for 4XX/5XX status codes
            
            data = response.json()
            items = data.get('message', {}).get('items', [])
            
            if not items:
                return None
                
            # Simple matching algorithm
            best_match = None
            best_score = 0
            
            for item in items:
                score = 0
                
                # Match title
                if title and item.get('title'):
                    item_title = item['title'][0] if isinstance(item['title'], list) and item['title'] else item.get('title', '')
                    title_similarity = self._text_similarity(title.lower(), str(item_title).lower())
                    score += title_similarity * 3
                
                # Match authors
                if authors and item.get('author'):
                    for ref_author in authors:
                        ref_last_name = ref_author.split()[-1].lower() if ref_author and ' ' in ref_author else ref_author.lower()
                        for item_author in item.get('author', []):
                            item_last_name = item_author.get('family', '').lower()
                            if ref_last_name and item_last_name and ref_last_name in item_last_name:
                                score += 1
                
                # Match year
                if year and item.get('published'):
                    published_parts = item.get('published', {}).get('date-parts', [[]])
                    if published_parts and published_parts[0] and str(year) == str(published_parts[0][0]):
                        score += 2
                
                if score > best_score:
                    best_score = score
                    best_match = item
            
            # Require a minimum score to consider it a match
            if best_score > 2:
                # Extract author names
                authors = []
                for author in best_match.get('author', []):
                    given = author.get('given', '')
                    family = author.get('family', '')
                    if given and family:
                        authors.append(f"{given} {family}")
                    elif family:
                        authors.append(family)
                
                # Extract publication date
                year = ""
                if best_match.get('published', {}).get('date-parts'):
                    date_parts = best_match['published']['date-parts'][0]
                    if date_parts and len(date_parts) > 0:
                        year = date_parts[0]
                
                return {
                    'title': best_match['title'][0] if isinstance(best_match.get('title'), list) and best_match['title'] else best_match.get('title', ''),
                    'authors': authors,
                    'year': year,
                    'journal': best_match.get('container-title', [''])[0] if isinstance(best_match.get('container-title'), list) else best_match.get('container-title', ''),
                    'url': best_match.get('URL', ''),
                    'doi': best_match.get('DOI', ''),
                    'abstract': ''  # CrossRef doesn't typically provide abstracts
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error searching CrossRef: {str(e)}")
            return None
            
    def search_researchgate(self, reference: Dict) -> Optional[Dict]:
        """
        Search ResearchGate for a reference using the OpenAI model as a proxy.
        
        Note: ResearchGate doesn't have a public API, so we use the AI model with internet access
        to simulate searching on ResearchGate.
        
        Args:
            reference: Dictionary with reference details
            
        Returns:
            Dictionary with paper details if found, None otherwise
        """
        # Create a formatted reference string
        ref_string = f"{', '.join(reference.get('authors', []))} ({reference.get('year', '')}). "
        ref_string += f"{reference.get('title', '')}. {reference.get('journal', '')}"
        
        prompt = f"""
        Search on ResearchGate for this academic reference:
        
        {ref_string}
        
        If you find the paper, provide the following information:
        1. Exact title
        2. Authors (full list)
        3. Year published
        4. Journal/venue
        5. DOI if available
        6. URL to the paper on ResearchGate
        7. Is the full text available?
        8. A short excerpt from the abstract
        
        Format the response as a JSON object with these fields. If you cannot find the exact paper, 
        return a JSON object with a field "found": false and explain why.
        """
        
        try:
            # This assumes the model has internet browsing capability
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # Use a model with internet browsing capability
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            if result.get('found') == False:
                return None
                
            return {
                'title': result.get('title'),
                'authors': result.get('authors'),
                'year': result.get('year'),
                'journal': result.get('journal'),
                'url': result.get('url'),
                'doi': result.get('doi'),
                'full_text_available': result.get('is_full_text_available'),
                'abstract': result.get('abstract_excerpt')
            }
        except Exception as e:
            logger.error(f"Error searching ResearchGate: {str(e)}")
            return None
    
    def verify_reference_exists(self, reference: Dict) -> Tuple[bool, Optional[Dict], str]:
        """Verify if a reference exists by searching multiple academic databases.
        
        Args:
            reference: Dictionary with reference details
            
        Returns:
            Tuple of (exists (bool), paper_details (Dict), source (str))
        """
        # Try Semantic Scholar first
        logger.info(f"Searching Semantic Scholar for: {reference.get('title')}")
        paper = self.search_semanticscholar(reference)
        if paper:
            return True, paper, "Semantic Scholar"
        
        # Try Google Scholar next
        logger.info(f"Searching Google Scholar for: {reference.get('title')}")
        paper = self.search_google_scholar(reference)
        if paper:
            return True, paper, "Google Scholar"
        
        # Try ResearchGate last
        logger.info(f"Searching ResearchGate for: {reference.get('title')}")
        paper = self.search_researchgate(reference)
        if paper:
            return True, paper, "ResearchGate"
        
        # Reference not found in any source
        return False, None, "Not found"
    
    def find_alternative_reference(self, claim: str, failed_reference: Dict) -> Optional[Dict]:
        """Find an alternative reference for a claim when the original reference is not found.
        
        Args:
            claim: The claim text that needs a reference
            failed_reference: The original reference that couldn't be found
            
        Returns:
            Dictionary with alternative reference details if found, None otherwise
        """
        prompt = f"""
        I need to find an alternative academic reference for this claim:
        
        "{claim}"
        
        The original reference was not found:
        {json.dumps(failed_reference, indent=2)}
        
        Please search academic sources to find a credible, verifiable reference that supports this claim.
        Prioritize recent papers (last 5 years) from reputable journals.
        Include full citation details: authors, title, year, journal, DOI, and URL if available.
        
        Format the response as a JSON object with these fields.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # Use a model with internet browsing capability
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate that we got a useful result
            if not result.get('title') or not result.get('authors'):
                return None
                
            return result
            
        except Exception as e:
            logger.error(f"Error finding alternative reference: {str(e)}")
            return None
    
    def verify_claim_with_reference(self, claim: str, reference_details: Dict) -> Tuple[bool, str]:
        """Verify if a claim is supported by the reference.
        
        Args:
            claim: The claim text to verify
            reference_details: Dictionary with paper details including abstract if available
            
        Returns:
            Tuple of (is_supported (bool), reasoning (str))
        """
        # Extract abstract or other content from the reference
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
        
        Evaluate if the claim is directly supported by the reference information. Consider:
        1. Does the claim align with what the paper is about?
        2. Is the claim a reasonable interpretation of the paper's findings?
        3. Does the abstract contain evidence supporting the claim?
        
        Format your response as a JSON object with these fields:
        - "is_supported": boolean (true if supported, false if not or unclear)
        - "confidence": number between 0-1
        - "reasoning": detailed explanation of your evaluation
        - "evidence": any specific text from the abstract that supports or contradicts the claim
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return (
                result.get('is_supported', False), 
                result.get('reasoning', 'No reasoning provided')
            )
            
        except Exception as e:
            logger.error(f"Error verifying claim: {str(e)}")
            return False, f"Error during verification: {str(e)}"
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple Jaccard similarity for word sets
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

def create_streamlit_app():
    """Create and run the Streamlit app for the reference verifier."""
    st.set_page_config(
        page_title="Academic Reference Verifier",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("Academic Reference Verification Agent")
    
    # Check if app is password protected
    try:
        openai_api_key, semanticscholar_api_key, app_password = load_api_keys()
        is_password_protected = app_password is not None
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        st.stop()
    
    # Authentication section
    authenticated = False
    
    if is_password_protected:
        st.sidebar.title("Authentication")
        password_input = st.sidebar.text_input("Enter password", type="password")
        
        if password_input:
            if password_input == app_password:
                authenticated = True
                st.sidebar.success("Authentication successful! ✅")
            else:
                st.sidebar.error("Incorrect password ❌")
    else:
        # No password required
        authenticated = True
    
    if not authenticated:
        st.info("Please enter the password in the sidebar to access this application.")
        st.stop()
    
    # Main application (only shown if authenticated)
    st.markdown("""
    This tool verifies academic references and claims in AI-generated content.
    
    1. Paste your AI-generated text with academic references
    2. The agent will verify if each reference exists
    3. For each claim, it will verify if the reference supports it
    4. If a reference doesn't exist or doesn't support a claim, alternative sources will be suggested
    """)
    
    # Initialize API clients after authentication
    openai.api_key = openai_api_key
    
    # Initialize Semantic Scholar only if API key is available
    ss = None
    if semanticscholar_api_key:
        try:
            ss = SemanticScholar(api_key=semanticscholar_api_key)
            logger.info("Semantic Scholar API initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize Semantic Scholar API: {str(e)}")
            st.sidebar.warning("Semantic Scholar API not available. The app will use other data sources.")
    else:
        logger.info("No Semantic Scholar API key provided, this data source will be skipped")
        st.sidebar.info("Semantic Scholar API key not provided. The app will use other data sources.")
    
    verifier = ReferenceVerifier()
    
    with st.form("reference_form"):
        ai_output = st.text_area(
            "Paste AI-generated text with academic references",
            height=300,
            help="Include the full text with claims and their citations."
        )
        
        submitted = st.form_submit_button("Verify References")
    
    if submitted and ai_output:
        with st.spinner("Extracting references and claims..."):
            references = verifier.extract_references(ai_output)
            claims = verifier.extract_claims_with_citations(ai_output)
            
            st.subheader("Extracted References")
            if references:
                for i, ref in enumerate(references):
                    st.write(f"{i+1}. {ref.get('title')} ({ref.get('year')})")
            else:
                st.warning("No references were extracted. Make sure your text includes properly formatted citations.")
            
            st.subheader("Extracted Claims")
            if claims:
                for i, claim_obj in enumerate(claims):
                    st.write(f"{i+1}. {claim_obj.get('claim')} [Citation: {claim_obj.get('citation')}]")
            else:
                st.warning("No claims with citations were extracted.")
        
        if references:
            st.subheader("Reference Verification Results")
            
            for i, ref in enumerate(references):
                with st.expander(f"Reference {i+1}: {ref.get('title')}", expanded=True):
                    with st.spinner(f"Verifying reference: {ref.get('title')}"):
                        exists, details, source = verifier.verify_reference_exists(ref)
                        
                        if exists:
                            st.success(f"✅ Reference found in {source}")
                            st.json(details)
                            
                            # Find claims that use this reference
                            matching_claims = []
                            for claim_obj in claims:
                                # This matching logic would need to be customized based on your citation style
                                citation = claim_obj.get('citation', '')
                                # Simple heuristic: if author name or year appears in the citation
                                if any(author.split()[-1] in citation for author in ref.get('authors', [])) or str(ref.get('year', '')) in citation:
                                    matching_claims.append(claim_obj.get('claim'))
                            
                            if matching_claims:
                                st.subheader("Claims supported by this reference:")
                                for j, claim_text in enumerate(matching_claims):
                                    with st.spinner(f"Verifying claim {j+1}"):
                                        is_supported, reasoning = verifier.verify_claim_with_reference(claim_text, details)
                                        
                                        if is_supported:
                                            st.success(f"✅ Claim {j+1}: {claim_text}")
                                            st.write(reasoning)
                                        else:
                                            st.error(f"❌ Claim {j+1}: {claim_text}")
                                            st.write(reasoning)
                                            
                                            with st.spinner("Finding alternative reference for this claim..."):
                                                alt_ref = verifier.find_alternative_reference(claim_text, ref)
                                                if alt_ref:
                                                    st.subheader("Alternative Reference Suggestion:")
                                                    st.json(alt_ref)
                                                    
                                                    # Verify the alternative reference
                                                    alt_exists, alt_details, alt_source = verifier.verify_reference_exists(alt_ref)
                                                    if alt_exists:
                                                        st.success(f"✅ Alternative reference verified in {alt_source}")
                                                        
                                                        # Verify if the alternative reference supports the claim
                                                        alt_supported, alt_reasoning = verifier.verify_claim_with_reference(claim_text, alt_details)
                                                        if alt_supported:
                                                            st.success("✅ Alternative reference supports the claim")
                                                            st.write(alt_reasoning)
                                                        else:
                                                            st.error("❌ Alternative reference does not support the claim")
                                                            st.write(alt_reasoning)
                                                    else:
                                                        st.error("❌ Alternative reference could not be verified")
                                                else:
                                                    st.error("❌ No suitable alternative reference found")
                            else:
                                st.info("No claims were found linked to this reference.")
                        else:
                            st.error("❌ Reference not found in any academic database")
                            
                            with st.spinner("Finding alternative reference..."):
                                # Use the first claim that might be related or create a generic query
                                search_text = claims[0].get('claim') if claims else f"Research about {ref.get('title')}"
                                alt_ref = verifier.find_alternative_reference(search_text, ref)
                                
                                if alt_ref:
                                    st.subheader("Alternative Reference Suggestion:")
                                    st.json(alt_ref)
                                    
                                    # Verify the alternative reference
                                    alt_exists, alt_details, alt_source = verifier.verify_reference_exists(alt_ref)
                                    if alt_exists:
                                        st.success(f"✅ Alternative reference verified in {alt_source}")
                                    else:
                                        st.error("❌ Alternative reference could not be verified")
                                else:
                                    st.error("❌ No suitable alternative reference found")

if __name__ == "__main__":
    create_streamlit_app()
