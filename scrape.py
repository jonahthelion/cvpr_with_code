import pandas as pd
import requests
from selenium import webdriver
from time import sleep
from tqdm import tqdm

def get_info(df, present_type):
    info = []
    for namei,name in enumerate(tqdm(df['Paper Title'])):
        success = False
        url = 'https://paperswithcode.com/search?q={}'.format(name.replace(' ', '+'))
        # hangs for some reason...
        if url == 'https://paperswithcode.com/search?q=Pluralistic+Image+Completion':
            info.append([name, present_type, 'https://github.com/lyndonzheng/Pluralistic-Inpainting', 132])
            continue
        driver.get(url)
        value = driver.find_element_by_class_name('infinite-container')
        items = value.find_elements_by_class_name('infinite-item')
        for item in items:
            title = item.find_element_by_tag_name('h1').text
            if title == name:
                badge = item.find_element_by_class_name('badge-dark')
                if badge.text == 'Code':
                    badge.click()
                    impls = driver.find_element_by_id('id_paper_implementations_collapsed')
                    github_link = impls.find_element_by_class_name('code-table-link').get_attribute('href')
                    stars = int(impls.text.split('\n')[1].replace(',', ''))
                    info.append([name, present_type, github_link, stars])
                    success = True
                break
        if not success:
            info.append([name, present_type, '', 0])
    return info

def get_score(row):
    score = row.stars
    if len(row['github link']) > 0:
        score += 0.5
    if row['presentation type'] == 'Oral':
        score += 0.1
    return score

driver = webdriver.Chrome('/Users/jonahphilion/Downloads/chromedriver')  # Optional argument, if not specified will search path.
info = []

df_oral = pd.read_csv('cvpr_2019_oral.csv')
df_poster = pd.read_csv('cvpr_2019_poster.csv')
df_poster = df_poster[~df_poster['Paper Title'].isin(set(df_oral['Paper Title']))]

info.extend(get_info(df_poster, 'Poster'))
info.extend(get_info(df_oral, 'Oral'))
driver.quit()

out_df = pd.DataFrame(info)
colnames = ['name', 'presentation type', 'github link', 'stars']
out_df.columns = colnames
out_df['score'] = out_df.apply(get_score, axis=1)
out_df = out_df.sort_values(['score', 'github link'], ascending=False)
out_df = out_df[colnames]
out_df.index = range(1, len(out_df) + 1)
outname = 'cvpr_2019_github_links.csv'
print('saving', outname)
out_df.to_csv(outname)


