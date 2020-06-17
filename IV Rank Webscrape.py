import requests
from bs4 import BeautifulSoup

URL = 'https://marketchameleon.com/volReports/VolatilityRankings'
page = requests.get(URL)

soup = BeautifulSoup(page.content, 'html.parser')
print(soup.prettify())
