import wikipediaapi
import re
import json
from wiki_articles_to_include import titles
from collections import Counter

wiki_wiki = wikipediaapi.Wikipedia('en')
texts = {}  # {city_name : tokenized_txt}
wiki_all_words_count = Counter()


def tokenize(text):  # taken from CS 4300
    return [x for x in re.findall(r"[a-z]+", text.lower()) if x != ""]


def scrape(wiki_all_words_count):
    for title in titles:
        page = wiki_wiki.page(title)
        txt = tokenize(page.text)
        print(title, len(txt))
        wiki_all_words_count += Counter(txt)
        texts[title] = txt


scrape(wiki_all_words_count)

culture_TO = tokenize(wiki_wiki.page('Culture in Toronto').text)

texts['Toronto'] = texts['Toronto'] + culture_TO
texts['Ithaca, New York'] += tokenize(
    wiki_wiki.page('Ithaca (town), New York').text)
texts['Ithaca, New York'] += tokenize(
    wiki_wiki.page('Cornell University').text)
texts['Buenos Aires'] += tokenize(wiki_wiki.page('Buenos Aires Province').text)
texts['Atlanta'] += tokenize(wiki_wiki.page('Atlanta metropolitan area').text)
texts['Madrid'] += tokenize(wiki_wiki.page('Spain').text)
texts['Tel Aviv'] += tokenize(wiki_wiki.page('Culture of Israel').text)

with open('wiki_texts.json', 'w') as f:
    json.dump(texts, f)
