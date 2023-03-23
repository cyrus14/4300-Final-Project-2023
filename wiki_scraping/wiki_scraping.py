import wikipediaapi
import re
import json
from wiki_articles_to_include import titles

wiki_wiki = wikipediaapi.Wikipedia('en')
texts = {} #{city_name : tokenized_txt}

def tokenize(text): #taken from CS 4300
    return [x for x in re.findall(r"[a-z]+", text.lower()) if x != ""]

for title in titles:
    page = wiki_wiki.page(title)
    txt = tokenize(page.text)
    texts[title] = txt

with open('wiki_texts.json', 'w') as f:
    json.dump(texts, f)