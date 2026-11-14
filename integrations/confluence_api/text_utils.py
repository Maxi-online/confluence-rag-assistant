from datetime import datetime, timezone, timedelta
from bs4 import BeautifulSoup

def format_date(date_str: str = None) -> str:
    'Returns the RFC3339 date for Moscow time (UTC+3)).'
    if not date_str:
        moscow_tz = timezone(timedelta(hours=3))
        return datetime.now(moscow_tz).replace(microsecond=0).isoformat()
    if date_str.endswith("Z"):
        date_str = date_str[:-1]
    return date_str

def static_clean_html(html: str) -> str:
    'Static function to clean the HTML code.'
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "ac:structured-macro", "ac:rich-text-body"]):
        tag.decompose()
    for table in soup.find_all("table"):
        table.unwrap()
    for header in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        header.insert_before("\n")
        header.unwrap()
    for li in soup.find_all("li"):
        li.insert_before("\n- ")
        li.unwrap()
    return soup.get_text(separator=" ", strip=True)
