######################################################
# Web scraper                                        #
######################################################

# Based on: 
# 1) https://www.freecodecamp.org/news/scraping-wikipedia-articles-with-python/
# 2) https://www.geeksforgeeks.org/web-scraping-from-wikipedia-using-python-a-complete-guide/

import requests
from bs4 import BeautifulSoup # beautifulsoup4
from typing import List

def scrape_wiki_text(url:str) -> List[str]:
    """Scrape a webpage from Wikipedia and return a list of the text (excluding HTML tags).

    Args:
        url (str): url to scrape from

    Raises:
        Exception: if response code is not 200 (OK). Read more here: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status#successful_responses

    Returns:
        List[str]: list of the text passages (no HTML tags)
    """
    # scrape webpage and check response code
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Response status code is not 200. It is:", response.status_code)
    
    # create soup object
    soup = BeautifulSoup(response.content, "html.parser")
    title = soup.find(id="firstHeading")
    print(f'Successfully scraped the webpage with the title: "{title.string}"')
    
    # get text extracts
    clean_texts = [text.get_text() for text in soup.find_all("p")] # wihout HTML tags
    return clean_texts

