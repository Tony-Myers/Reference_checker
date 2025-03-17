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
        self.ss = semantic_scholar_client  # This might be None if the API key is unavailable
    
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
                authors = []
                for author in best_match.get('author', []):
                    given = author.get('given', '')
                    family = author.get('family', '')
                    if given and family:
                        authors.append(f"{given} {family}")
                    elif family:
                        authors.append(family)
                
                # Extract publication date safely
                year = ""
                if best_match.get('published', {}).get('date-parts'):
                    date_parts = best_match['published']['date-parts'][0]
                    if date_parts and len(date_parts) > 0:
                        year = date_parts[0]
                
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
                    'authors': authors,
                    'year': year,
                    'journal': journal,
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
        
        # Try CrossRef (reliable and doesn't require API key)
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
                    
                    # If there's reasonable similarity, consider it a match
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
            failed_reference: The original reference that couldn't be found
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
                    logger.warning(f"Attempt {attempt+1}: Got incomplete reference data")
                    continue
                
                # Verify this alternative reference actually exists
                exists, details, source = self.advanced_verify_reference_exists(result)
                if exists:
                    logger.info(f"Found verified alternative reference on attempt {attempt+1}")
                    return result
                
                logger.warning(f"Attempt {attempt+1}: Found reference but couldn't verify it exists")
                
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
