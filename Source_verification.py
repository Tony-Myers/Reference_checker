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
    
    def __init__(self, semantic_scholar_client=None):
        self.openai_client = openai.OpenAI(api_key=openai.api_key)
        self.ss = semantic_scholar_client

    def extract_references(self, ai_output: str) -> List[Dict]:
        """Extract academic references from an AI-generated output."""
    prompt = f'''
        Please extract all academic references from the following text. Note that references might appear as numeric citations in the text (e.g. "1–5") and as a separate, numbered reference list. For any numeric range (e.g. "1–5"), assume that each number in the range corresponds to a distinct reference and expand them accordingly. 
            
        For each reference, extract the following details:
        1. Reference number (if available)
        2. Authors (as a list of names)
        3. Title
        4. Year
        5. Journal, Conference, or Publisher
        6. DOI (if available)
        7. URL (if available)

        Return the output as a JSON array of objects, where each object corresponds to one reference with the above fields. For any missing information, use an empty string or null.

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
                matches = re.findall(r'\[.*\]', response.choices[0].message.content, re.DOTALL)
                if matches:
                    try:
                        return json.loads(matches[0])
                    except Exception:
                        logger.error("Failed to parse references from extracted JSON array")
                return []
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from response")
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
                # Handle case where the AI did not use the expected structure
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
                    # Safe handling of title list
                    if isinstance(item.get('title'), list) and len(item.get('title')) > 0:
                        item_title = item['title'][0]
                    else:
                        item_title = str(item.get('title', ''))
                    
                    title_similarity = self._text_similarity(title.lower(), item_title.lower())
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
                    if published_parts and len(published_parts) > 0 and len(published_parts[0]) > 0:
                        pub_year = published_parts[0][0]
                        if str(year) == str(pub_year):
                            score += 2
                
                if score > best_score:
                    best_score = score
                    best_match = item
            
            # Require a minimum score to consider it a match
            if best_score > 2 and best_match:
                # Extract author names safely
                authors_list = []
                for author in best_match.get('author', []):
                    given = author.get('given', '')
                    family = author.get('family', '')
                    if given and family:
                        authors_list.append(f"{given} {family}")
                    elif family:
                        authors_list.append(family)
                
                # Extract publication date safely
                pub_year = ""
                if best_match.get('published', {}).get('date-parts'):
                    date_parts = best_match['published']['date-parts'][0]
                    if date_parts and len(date_parts) > 0:
                        pub_year = date_parts[0]
                
                # Get title safely
                if isinstance(best_match.get('title'), list) and len(best_match.get('title', [])) > 0:
                    title = best_match['title'][0]
                else:
                    title = str(best_match.get('title', ''))
                    
                # Get journal safely
                journal = ""
                if isinstance(best_match.get('container-title'), list) and len(best_match.get('container-title', [])) > 0:
                    journal = best_match['container-title'][0]
                else:
                    journal = str(best_match.get('container-title', ''))
                
                return {
                    'title': title,
                    'authors': authors_list,
                    'year': pub_year,
                    'journal': journal,
                    'url': best_match.get('URL', ''),
                    'doi': best_match.get('DOI', ''),
                    'abstract': ''  # CrossRef does not typically provide abstracts
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error searching CrossRef: {str(e)}")
            return None
            
    def search_researchgate(self, reference: Dict) -> Optional[Dict]:
        """
        Search ResearchGate for a reference using the OpenAI model as a proxy.
        
        Note: ResearchGate does not have a public API, so we use the AI model with internet access
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

    def search_google(self, reference: Dict) -> Optional[Dict]:
        """Search Google for a reference using the OpenAI model as a proxy.
        
        Args:
            reference: Dictionary with reference details
            
        Returns:
            Dictionary with paper details if found, None otherwise
        """
        # Create a formatted reference string
        authors_str = ', '.join(reference.get('authors', []))
        title = reference.get('title', '')
        year = reference.get('year', '')
        journal = reference.get('journal', '')
        
        # Construct a specific search query for academic papers
        search_query = f'"{title}"'
        if authors_str:
            search_query += f' author:{authors_str}'
        if year:
            search_query += f' {year}'
        if journal:
            search_query += f' {journal}'
        
        # Add academic-specific terms to improve results
        search_query += ' filetype:pdf OR site:scholar.google.com OR site:researchgate.net OR site:academia.edu OR site:sciencedirect.com'
        
        prompt = f"""
        Please perform a Google search for this academic reference:
        
        Search query: {search_query}
        
        Reference details:
        - Title: {title}
        - Authors: {authors_str}
        - Year: {year}
        - Journal: {journal}
        
        If you find the paper, provide the following information:
        1. Exact title
        2. Authors (full list)
        3. Year published
        4. Journal/venue
        5. DOI if available
        6. URL to the paper
        7. Is the full text available?
        8. A short excerpt from the abstract if available
        
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
                'authors': result.get('authors', []),
                'year': result.get('year'),
                'journal': result.get('journal'),
                'url': result.get('url'),
                'doi': result.get('doi'),
                'full_text_available': result.get('full_text_available', False),
                'abstract': result.get('abstract', '')
            }
        except Exception as e:
            logger.error(f"Error searching Google: {str(e)}")
            return None

    def search_with_doi(self, doi: str) -> Optional[Dict]:
        """Search for a reference using its DOI.
        
        Args:
            doi: DOI string
        
        Returns:
            Dictionary with paper details if found, None otherwise
        """
        if not doi:
            return None
            
        try:
            # Try CrossRef's DOI API
            response = requests.get(f'https://api.crossref.org/works/{doi}')
            if response.status_code == 200:
                data = response.json().get('message', {})
                
                # Extract author names
                authors = []
                for author in data.get('author', []):
                    given = author.get('given', '')
                    family = author.get('family', '')
                    if given and family:
                        authors.append(f"{given} {family}")
                    elif family:
                        authors.append(family)
                
                # Extract year
                year = ""
                if data.get('published', {}).get('date-parts'):
                    date_parts = data['published']['date-parts'][0]
                    if date_parts and len(date_parts) > 0:
                        year = date_parts[0]
                
                # Get title
                title = ""
                if isinstance(data.get('title'), list) and len(data.get('title', [])) > 0:
                    title = data['title'][0]
                
                # Get journal
                journal = ""
                if isinstance(data.get('container-title'), list) and len(data.get('container-title', [])) > 0:
                    journal = data['container-title'][0]
                
                return {
                    'title': title,
                    'authors': authors,
                    'year': year,
                    'journal': journal,
                    'url': data.get('URL', ''),
                    'doi': doi,
                    'abstract': ''
                }
            return None
        except Exception as e:
            logger.error(f"Error searching by DOI: {str(e)}")
            return None

    def search_google_scholar(self, reference: Dict) -> Optional[Dict]:
        """Search Google Scholar for a reference using the OpenAI model.
        
        Args:
            reference: Dictionary with reference details
            
        Returns:
            Dictionary with paper details if found, None otherwise
        """
        # Create search query specifically for Google Scholar
        authors_str = ' '.join([a.split()[-1] for a in reference.get('authors', []) if a])  # Last names only
        title = reference.get('title', '')
        year = reference.get('year', '')
        
        prompt = f"""
        Please search Google Scholar for this academic reference:
        
        Title: {title}
        Authors: {authors_str}
        Year: {year}
        
        Use Google Scholar (scholar.google.com) specifically. If you find the paper, provide:
        1. Complete title
        2. Complete list of authors
        3. Year published
        4. Journal/venue
        5. DOI if shown
        6. Citation count if available
        7. Link to the paper
        8. Brief excerpt from the abstract or summary
        
        Format as a JSON object. If you cannot find the paper, return a JSON with "found": false.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            if result.get('found') == False:
                return None
                
            return {
                'title': result.get('title'),
                'authors': result.get('authors', []),
                'year': result.get('year'),
                'journal': result.get('journal'),
                'url': result.get('url'),
                'doi': result.get('doi'),
                'citation_count': result.get('citation_count'),
                'abstract': result.get('abstract_excerpt', '')
            }
        except Exception as e:
            logger.error(f"Error searching Google Scholar: {str(e)}")
            return None
    
    def verify_reference_exists(self, reference: Dict) -> Tuple[bool, Optional[Dict], str]:
        """Verify if a reference exists by searching multiple academic databases.
        
        Args:
            reference: Dictionary with reference details
            
        Returns:
            Tuple of (exists (bool), paper_details (Dict), source (str))
        """
        search_results = []
        sources = []
        
        # Check if Semantic Scholar is available
        if hasattr(self, 'ss') and self.ss is not None:
            logger.info(f"Searching Semantic Scholar for: {reference.get('title')}")
            try:
                paper = self.search_semanticscholar(reference)
                if paper:
                    search_results.append(paper)
                    sources.append("Semantic Scholar")
            except Exception as e:
                logger.warning(f"Error searching Semantic Scholar: {str(e)}")
        else:
            logger.warning("Semantic Scholar API not initialized, skipping this source")
        
        # Try CrossRef (reliable and does not require API key)
        logger.info(f"Searching CrossRef for: {reference.get('title')}")
        try:
            paper = self.search_crossref(reference)
            if paper:
                search_results.append(paper)
                sources.append("CrossRef")
        except Exception as e:
            logger.warning(f"Error searching CrossRef: {str(e)}")
        
        # Try ResearchGate
        logger.info(f"Searching ResearchGate for: {reference.get('title')}")
        try:
            paper = self.search_researchgate(reference)
            if paper:
                search_results.append(paper)
                sources.append("ResearchGate")
        except Exception as e:
            logger.warning(f"Error searching ResearchGate: {str(e)}")
        
        # Try general Google search
        logger.info(f"Trying Google search for: {reference.get('title')}")
        try:
            paper = self.search_google(reference)
            if paper:
                search_results.append(paper)
                sources.append("Google Search")
        except Exception as e:
            logger.warning(f"Error searching Google: {str(e)}")
    
        # Return the first valid result if any are found
        if search_results:
            return True, search_results[0], sources[0]
        
        # Reference not found in any source
        return False, None, "Not found"

    def advanced_verify_reference_exists(self, reference: Dict) -> Tuple[bool, Optional[Dict], str]:
        """Enhanced verification with multiple strategies and fallbacks.
        
        Args:
            reference: Dictionary with reference details
            
        Returns:
            Tuple of (exists (bool), paper_details (Dict), source (str))
        """
        # Check if DOI is available and use it directly
        if reference.get('doi'):
            logger.info(f"Searching by DOI: {reference.get('doi')}")
            try:
                paper = self.search_with_doi(reference.get('doi'))
                if paper:
                    return True, paper, "DOI Lookup"
            except Exception as e:
                logger.warning(f"Error searching by DOI: {str(e)}")
        
        # Standard reference verification (existing method)
        exists, details, source = self.verify_reference_exists(reference)
        if exists:
            return True, details, source
        
        # If title has very weird formatting, try normalizing it
        if not exists and reference.get('title'):
            normalized_title = re.sub(r'[^\w\s]', '', reference.get('title')).lower()
            if normalized_title != reference.get('title').lower():
                logger.info(f"Trying with normalized title: {normalized_title}")
                normalized_ref = reference.copy()
                normalized_ref['title'] = normalized_title
                exists, details, source = self.verify_reference_exists(normalized_ref)
                if exists:
                    return True, details, source + " (with normalized title)"
        
        # Try Google Scholar as a deeper fallback
        if not exists:
            logger.info(f"Trying Google Scholar for: {reference.get('title')}")
            try:
                paper = self.search_google_scholar(reference)
                if paper:
                    return True, paper, "Google Scholar"
            except Exception as e:
                logger.warning(f"Error searching Google Scholar: {str(e)}")
        
        # If we have authors and year but no match, try a broader search
        if not exists and reference.get('authors') and reference.get('year'):
            # Use just the first author's last name and year
            first_author = reference.get('authors')[0].split()[-1]
            year = reference.get('year')
            
            logger.info(f"Trying broader search with first author and year: {first_author} {year}")
            
            prompt = f"""
            Please perform a comprehensive search for this academic reference:
            
            First author's last name: {first_author}
            Year: {year}
            
            Check Google Scholar, Semantic Scholar, ResearchGate, Academia.edu, 
            university repositories, and other academic sources.
            
            If you find a paper that could be a match for "{reference.get('title')}", provide:
            1. Title
            2. Authors
            3. Year
            4. Journal/venue
            5. DOI or URL
            6. Brief description of the content
            
            Format as a JSON object. If you cannot find anything that could be a match, return a JSON with "found": false.
            """
            
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                
                if result.get('found') != False and result.get('title'):
                    # Calculate similarity to the original title
                    original_title = reference.get('title', '')
                    found_title = result.get('title', '')
                    similarity = self._text_similarity(original_title, found_title)
                    
                    # If there is reasonable similarity, consider it a match
                    if similarity > 0.4:  # Adjust threshold as needed
                        logger.info(f"Found possible match with similarity {similarity}")
                        return True, result, "Broad Academic Search"
            except Exception as e:
                logger.warning(f"Error in broader search: {str(e)}")
        
        # Reference not found in any source
        return False, None, "Not found"
    
    def find_alternative_reference(self, claim: str, failed_reference: Dict, 
                                   max_attempts: int = 4) -> Optional[Dict]:
        """Find an alternative reference for a claim when the original reference is not found.
        
        Args:
            claim: The claim text that needs a reference
            failed_reference: The original reference that could not be found
            max_attempts: Maximum number of attempts to find an alternative
            
        Returns:
            Dictionary with alternative reference details if found, None otherwise
        """
        for attempt in range(max_attempts):
            logger.info(f"Attempt {attempt+1} of {max_attempts} to find alternative reference")
            
            # Add information about previous attempts in later iterations
            previous_attempts_text = ""
            if attempt > 0:
                previous_attempts_text = f"\nThis is attempt {attempt+1} of {max_attempts}. "
                previous_attempts_text += "Please try different search terms, journals, or authors than previous attempts."
            
            prompt = f"""
            I need to find an alternative academic reference for this claim:
            
            "{claim}"
            
            The original reference was not found:
            {json.dumps(failed_reference, indent=2)}
            {previous_attempts_text}
            
            Please search academic sources to find a credible, verifiable reference that supports this claim.
            Prioritise recent papers (last 5 years) from reputable journals.
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
                    logger.warning(f"Attempt {attempt+1}: Got incomplete reference data")
                    continue
                
                # Verify this alternative reference actually exists
                exists, details, source = self.advanced_verify_reference_exists(result)
                if exists:
                    logger.info(f"Found verified alternative reference on attempt {attempt+1}")
                    return result
                
                logger.warning(f"Attempt {attempt+1}: Found reference but could not verify it exists")
                
            except Exception as e:
                logger.error(f"Error finding alternative reference (attempt {attempt+1}): {str(e)}")
            
            # Short delay between attempts to avoid rate limiting
            time.sleep(1)
                
        logger.warning("All attempts to find alternative reference failed")
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
    
    def format_apa_reference(self, reference: Dict) -> str:
        """Format reference details into APA style citation.
        
        Args:
            reference: Dictionary with reference details
            
        Returns:
            String with APA formatted reference
        """
        try:
            # Extract required components
            authors = reference.get('authors', [])
            year = reference.get('year', '')
            title = reference.get('title', '')
            journal = reference.get('journal', '')
            doi = reference.get('doi', '')
            url = reference.get('url', '')
            
            # Format authors (Last, F. I., & Last, F. I.)
            formatted_authors = ""
            if authors:
                author_list = []
                for author in authors:
                    parts = author.split()
                    if len(parts) > 1:
                        # Last name then initials
                        last = parts[-1]
                        initials = ''.join([p[0] + '.' for p in parts[:-1]])
                        author_list.append(f"{last}, {initials}")
                    else:
                        # Just use the whole name if parsing fails
                        author_list.append(author)
                
                if len(author_list) == 1:
                    formatted_authors = author_list[0]
                elif len(author_list) == 2:
                    formatted_authors = f"{author_list[0]} & {author_list[1]}"
                else:
                    formatted_authors = ", ".join(author_list[:-1]) + ", & " + author_list[-1]
            
            # Format title (only capitalise first word and proper nouns)
            formatted_title = title
            
            # Format journal (in italics - using markdown)
            formatted_journal = f"*{journal}*" if journal else ""
            
            # Format DOI or URL if available
            doi_or_url = ""
            if doi:
                doi_or_url = f"https://doi.org/{doi}"
            elif url:
                doi_or_url = url
            
            # Assemble the APA reference
            apa_reference = f"{formatted_authors} ({year}). {formatted_title}. "
            if formatted_journal:
                apa_reference += f"{formatted_journal}. "
            if doi_or_url:
                apa_reference += f"{doi_or_url}"
            
            return apa_reference
            
        except Exception as e:
            logger.error(f"Error formatting APA reference: {str(e)}")
            # Return basic citation if formatting fails
            return f"{', '.join(reference.get('authors', []))} ({reference.get('year', '')}). {reference.get('title', '')}."
    
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
    This tool verifies academic references and claims in AI-generated content using multiple search strategies:

    1. Paste your AI-generated text with academic references
    2. The agent will verify if each reference exists using:
       - Direct DOI lookup
       - Semantic Scholar (if available)
       - CrossRef
       - ResearchGate
       - Google Search
       - Google Scholar
    3. For each claim, it will verify if the reference supports it
    4. If a reference does not exist or does not support a claim, alternative sources will be suggested
    """)
    
    # Add search settings in the sidebar
    st.sidebar.title("Search Settings")
    search_thoroughness = st.sidebar.select_slider(
        "Search thoroughness",
        options=["Fast", "Standard", "Thorough"],
        value="Standard",
        help="Fast: Quick searches only. Standard: Regular search strategy. Thorough: Try all possible search methods (slower but more comprehensive)"
    )
    
    # Initialize API clients after authentication
    openai.api_key = openai_api_key
    
    # Initialize Semantic Scholar only if API key is available
    semantic_scholar_client = None
    if semanticscholar_api_key:
        try:
            semantic_scholar_client = SemanticScholar(api_key=semanticscholar_api_key)
            logger.info("Semantic Scholar API initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize Semantic Scholar API: {str(e)}")
            st.sidebar.warning("Semantic Scholar API not available. The app will use other data sources.")
    else:
        logger.info("No Semantic Scholar API key provided, this data source will be skipped")
        st.sidebar.info("Semantic Scholar API key not provided. The app will use other data sources.")
    
    verifier = ReferenceVerifier(semantic_scholar_client=semantic_scholar_client)
    
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
                        # Choose verification method based on the selected thoroughness level
                        if search_thoroughness == "Fast":
                            # Use CrossRef only for fastest results
                            paper = verifier.search_crossref(ref)
                            exists = paper is not None
                            details = paper
                            source = "CrossRef"
                        elif search_thoroughness == "Thorough":
                            # Use the most comprehensive search
                            exists, details, source = verifier.advanced_verify_reference_exists(ref)
                        else:
                            # Standard search
                            exists, details, source = verifier.verify_reference_exists(ref)
                        
                        if exists:
                            st.success(f"✅ Reference found in {source}")
                            # Add info about which search method was successful
                            st.info(f"Search method used: {source}")
                            
                            # Generate and display APA citation
                            apa_citation = verifier.format_apa_reference(details)
                            st.subheader("APA Citation:")
                            st.markdown(apa_citation)
                            
                            # Display DOI if available
                            if details.get('doi'):
                                st.subheader("DOI:")
                                st.markdown(f"[{details['doi']}](https://doi.org/{details['doi']})")
                            
                            # Show reference details directly (without an expander when inside another expander)
                            st.subheader("Reference Details:")
                            st.json(details)
                            
                            # Find claims that use this reference
                            matching_claims = []
                            for claim_obj in claims:
                                # This matching logic would need to be customised based on your citation style
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
                                                alt_ref = verifier.find_alternative_reference(claim_text, ref, max_attempts=4)
                                                if alt_ref:
                                                    st.subheader("Alternative Reference Suggestion:")
                                                    
                                                    # Verify the alternative reference
                                                    # Use advanced verification for thorough searches
                                                    if search_thoroughness == "Thorough":
                                                        alt_exists, alt_details, alt_source = verifier.advanced_verify_reference_exists(alt_ref)
                                                    else:
                                                        alt_exists, alt_details, alt_source = verifier.verify_reference_exists(alt_ref)
                                                    
                                                    if alt_exists:
                                                        st.success(f"✅ Alternative reference verified in {alt_source}")
                                                        st.info(f"Search method used: {alt_source}")
                                                        
                                                        # Show APA citation for alternative reference
                                                        alt_apa_citation = verifier.format_apa_reference(alt_details)
                                                        st.markdown(alt_apa_citation)
                                                        
                                                        # Display DOI if available
                                                        if alt_details.get('doi'):
                                                            st.markdown(f"**DOI:** [{alt_details['doi']}](https://doi.org/{alt_details['doi']})")
                                                        
                                                        # Show reference details directly
                                                        st.subheader("Alternative Reference Details:")
                                                        st.json(alt_details)
                                                        
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
                                                        st.json(alt_ref)
                                                else:
                                                    st.error("❌ No suitable alternative reference found after multiple attempts")
                            else:
                                st.info("No claims were found linked to this reference.")
                        else:
                            st.error("❌ Reference not found in any academic database")
                            
                            # Try advanced search if not already used and not fast mode
                            if search_thoroughness != "Fast" and search_thoroughness != "Thorough":
                                st.warning("Attempting advanced search strategies...")
                                exists, details, source = verifier.advanced_verify_reference_exists(ref)
                                if exists:
                                    st.success(f"✅ Reference found with advanced search in {source}")
                                    st.info(f"Advanced search method used: {source}")
                                    
                                    # Display the same details as above
                                    apa_citation = verifier.format_apa_reference(details)
                                    st.subheader("APA Citation:")
                                    st.markdown(apa_citation)
                                    
                                    if details.get('doi'):
                                        st.subheader("DOI:")
                                        st.markdown(f"[{details['doi']}](https://doi.org/{details['doi']})")
                                    
                                    st.subheader("Reference Details:")
                                    st.json(details)
                                else:
                                    # If still not found after trying advanced search
                                    with st.spinner("Finding alternative reference..."):
                                        # Use the first claim that might be related or create a generic query
                                        search_text = claims[0].get('claim') if claims else f"Research about {ref.get('title')}"
                                        alt_ref = verifier.find_alternative_reference(search_text, ref, max_attempts=4)
                                        
                                        if alt_ref:
                                            st.subheader("Alternative Reference Suggestion:")
                                            
                                            # Verify the alternative reference with appropriate strategy
                                            if search_thoroughness == "Thorough":
                                                alt_exists, alt_details, alt_source = verifier.advanced_verify_reference_exists(alt_ref)
                                            else:
                                                alt_exists, alt_details, alt_source = verifier.verify_reference_exists(alt_ref)
                                                
                                            if alt_exists:
                                                st.success(f"✅ Alternative reference verified in {alt_source}")
                                                st.info(f"Search method used: {alt_source}")
                                                
                                                # Show APA citation for alternative reference
                                                alt_apa_citation = verifier.format_apa_reference(alt_details)
                                                st.markdown(alt_apa_citation)
                                                
                                                # Display DOI if available
                                                if alt_details.get('doi'):
                                                    st.markdown(f"**DOI:** [{alt_details['doi']}](https://doi.org/{alt_details['doi']})")
                                                
                                                # Show reference details directly
                                                st.subheader("Alternative Reference Details:")
                                                st.json(alt_details)
                                            else:
                                                st.error("❌ Alternative reference could not be verified")
                                                st.json(alt_ref)
                                        else:
                                            st.error("❌ No suitable alternative reference found after multiple attempts")


if __name__ == "__main__":
    create_streamlit_app()
