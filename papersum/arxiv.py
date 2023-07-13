import os
import re
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from urllib.parse import quote
from PyPDF2 import PdfReader
def fetch_arxiv_papers(topic: str, days_ago: int = 2, maxpapers: int = 4):
    """
    Fetches arxiv papers for a given topic and date range.
    Parameters
    ----------
    topic : str
        The topic to search for.
    days_ago: int
        The number of days to look for in the past
    maxpapers : int
        The maximum number of papers to fetch.
    Returns
    -------
    papers : dict
        A dictionary of papers, with the title as the key and the text as the value.
    Examples
    --------
    papers = fetch_arxiv_papers("neural network potentials", 2, 4)
    for title, text in papers.items():
      print(f"Title: {title}")
      print(f"Text: {text[:100]}...")  # Printing only first 100 characters of the paper

    """
    datebeg = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    dateend = datetime.now().strftime("%Y-%m-%d")
    topic = quote(topic)

    os.makedirs("input", exist_ok=True)

    search_url = f"https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term={topic}&terms-0-field=all&classification-physics_archives=all&classification-include_cross_list=include&date-year=&date-filter_by=date_range&date-from_date={datebeg}&date-to_date={dateend}&date-date_type=submitted_date&abstracts=show&size=50&order=-announced_date_first"
    response = requests.get(search_url)
    assert response.status_code == 200

    pdf_urls = re.findall(r"https://arxiv.org/pdf/[0-9]+\.[0-9]+", response.text)[
        :maxpapers
    ]

    papers = {}
    for url in pdf_urls:
        doi = re.search(r"[0-9]+\.[0-9]+", url).group()
        paper_page = requests.get(f"https://arxiv.org/abs/{doi}")
        soup = BeautifulSoup(paper_page.text, "html.parser")
        title = soup.title.string.split("]")[1].strip()
        print(f"Getting {doi} : {title}")
        filename = f"input/{doi}.pdf"
        #Do not download if already downloaded
        if not os.path.exists(filename):
            paper = requests.get(f"{url}.pdf")
            with open(filename, "wb") as f:
                f.write(paper.content)

        # with pdfplumber.open(filename) as pdf:
        #     paper_text = "\n".join(page.extract_text(x_tolerance=1) for page in pdf.pages)
        with open(filename, 'rb') as f:
            pdf = PdfReader(f)
            paper_text = ''
            for page in pdf.pages:
                paper_text += page.extract_text()
        papers[title] = paper_text

    return papers
