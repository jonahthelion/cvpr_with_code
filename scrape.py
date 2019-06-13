import pandas as pd
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from multiprocessing import Pool

def get_score(row):
    score = row.stars if not row.stars == '' else 0
    if len(row['github']) > 0:
        score += 0.5
    if len(row['Oral day']) > 0:
        score += 0.1
    return score

def get_paper_info():
    print('getting paper info...')
    info = {}
    base_url = 'http://openaccess.thecvf.com/CVPR2019.py'
    soup = BeautifulSoup(requests.get(base_url).content, "lxml")
    for a in tqdm(soup.find_all('a')):
        link = a.get('href')
        if link is None or link[-5:] != '.html':
            continue
        link = "http://openaccess.thecvf.com/{}".format(a.get('href'))
        # bug in cvf
        if link == "http://openaccess.thecvf.com/content_CVPR_2019/html/Gao_2.5D_Visual_Sound_CVPR_2019_paper.html":
            link = "http://openaccess.thecvf.com/content_CVPR_2019/html/Gao_2.html"
        elif link == "http://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_AET_vs._AED_Unsupervised_Representation_Learning_by_Auto-Encoding_Transformations_Rather_CVPR_2019_paper.html":
            link = "http://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_AET_vs.html"
        paper = BeautifulSoup(requests.get(link).content, "lxml")
        title = unidecode(paper.find('div', {'id': 'papertitle'}).text.lstrip())
        authors = unidecode(paper.find('i').text).split(',  ')
        abstract = unidecode(paper.find('div', {'id': 'abstract'}).text.strip())
        info[title.lower()] = {'authors': authors, 'abstract': abstract, 'title': title}
    return info

def get_conference_info():
    print('getting conference info...')
    url = 'http://cvpr2019.thecvf.com/program/main_conference'
    soup = BeautifulSoup(requests.get(url).content, 'lxml')
    ix2conf = []
    for t in soup.find_all('h4'):
        date, ptype, _ = unidecode(t.text).split(' &nbsp ')
        ix2conf.append({'day': date.split(',')[0], 'ptype': ptype[:-2]})
    info = {}
    for i, tab in enumerate(tqdm(soup.find_all('table'))):
        for row in tab.find_all('tr'):
            tds = row.find_all('td')
            if len(tds) != 6:
                continue
            title = unidecode(tds[3].text).lower()
            if not title in info:
                info[title] = {}
            info[title]['Poster #'] = int(unidecode(tds[1].text))
            ptype = ix2conf[i]['ptype'].split(' ')[0]
            info[title]['{} day'.format(ptype)] = ix2conf[i]['day']
            info[title]['{} session'.format(ptype)] = ix2conf[i]['ptype']
            if ptype == 'Oral':
                info[title]['Oral time'] = unidecode(tds[2].text)
    return info

def get_row(row):
    return {'Poster #': row['Poster #'],
        'Poster day': row['Poster day'],
        'Poster session': row['Poster session'],
        'First author': row['authors'][0],
        'github': row['github'] if 'github' in row else '',
        'stars': row['stars'] if 'stars' in row else '',
        'title': row['title'],
        'Oral day': row['Oral day'] if 'Oral day' in row else '',
        'Oral session': row['Oral session'] if 'Oral session' in row else '',
        'Oral time': row['Oral time'] if 'Oral time' in row else '',
        }

def get_github(rowi):
    row = info[rowi]
    url = 'https://paperswithcode.com/search?q={}'.format(row['title'].replace(' ', '+'))
    results = BeautifulSoup(requests.get(url).content, 'lxml')
    has_code = []
    for res in results.find_all('div', {'class': 'infinite-item'}):
        link = res.find('a', {'class': 'badge-dark'})
        if unidecode(link.text.strip()) == 'Code':
            has_code.append('https://paperswithcode.com{}'.format(link.get('href')))
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([row['abstract']]).A
    Y = []
    papers = []
    for link in has_code:
        paper = BeautifulSoup(requests.get(link).content, 'lxml')
        papers.append(paper)
        paperabs = unidecode(paper.find('div', {'class': 'paper-abstract'}).find('p').text.strip().replace('\n', ' ').replace('...', ' ').replace(' (read more)', ''))
        Y.append(paperabs)
    if len(Y) > 0:
        Y = vectorizer.transform(Y).A
        scores = np.matmul(Y, X.T)[:,0]
        best_ix = np.argmax(scores)
        if scores[best_ix] > .85:
            paper = papers[best_ix]
            code = paper.find('div', {'class': 'paper-implementations'}).find('div', {'class': 'row'})
            github = code.find('a', {'class': 'code-table-link'}).get('href')
            stars = int(unidecode(code.find_all('div', {'class': 'paper-impl-cell'})[1].text).strip().replace(' ', '').replace(',', ''))
            return rowi, {'github': github, 'stars': stars}
    return rowi, {}

info = get_paper_info()
conference_info = get_conference_info()
for k in info:
    info[k].update(conference_info[k])

print('getting github info...')
po = Pool(4)
github_info = po.map(get_github, info.keys())
for k, val in github_info:
    info[k].update(val)
            
info = map(get_row, info.values())
df = pd.DataFrame(info)
df['score'] = df.apply(get_score, axis=1)
df = df.sort_values(['score', 'github'], ascending=False)
df = df[['title', 'stars', 'github', 'Oral day', 'Oral session', 'Oral time', 'Poster day', 'Poster session', 'Poster #', 'First author']]
df.index = range(1, len(df) + 1)
outname = 'cvpr_2019_github_links.csv'
print('saving', outname)
df.to_csv(outname)