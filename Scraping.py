import csv
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup

## To get all US cities
cities_list = 'https://www.craigslist.org/about/sites#US'
uClient = uReq(cities_list)
cities_html = uClient.read()
uClient.close()
cites_soup = soup(cities_html, "html.parser")
cites_li = cites_soup.findAll("li")
city_url_list = []
cities =[]
#Appending all the cities to a list
for a in cites_li:
  chk= a.text
  cities.append(chk)


#To get all the links available in the cities if multiple pages
list_of_cities = []
for city in cities:
    page_number = 0
    #Since majority of the posts are there in only one page
    while page_number < 2:
        city_link = "https://"+ str(city) + ".craigslist.org/search/edu#search=1~list~" +str(page_number)+ "~0"
        list_of_cities.append(city_link)
        page_number +=120


#extract and store each education posting
links = []
# URLs counter
import requests
from random import randint
import time
from selenium import webdriver
from bs4 import BeautifulSoup

job_urls = 1
for each_city_page in list_of_cities:
    #Enabled Java-extension and used selenium in extracting the html links of each posts
    browser = webdriver.Chrome('chromedriver')
    options = webdriver.ChromeOptions()
    options.headless = True
    options.add_argument('--enable-javascript')
    options.add_argument("--headless")

    try:
        browser.get(each_city_page)
        #giving time,.sleep after the pinging of website to avoid getting blocked
        time.sleep(2)
        html = browser.page_source
        soup = BeautifulSoup(html, 'html.parser')
        #Using macro container for that page in education subcategory
        posts = soup.find_all('a', class_="titlestring")
        # getting all the html links in the page and appending them to a list
        for link in posts:
            l = link.get('href')
            links.append(l)

    except:
        pass
    time.sleep(randint(0, 1))
    job_urls += 1


import pandas as pd
#df = pd.DataFrame(links)
#df.to_csv("./links_sf.csv", sep=',',index=False)

#links = pd.read_csv('./links_sf_etc.csv',names=['https'])
#links = list(links['https'][1:])

import re
count = 0
jobs = []
# looping over all links in the lists list
for link in links:
    #Using BS4 to get the requests instead of selenium for more faster results
    each_page = requests.get(link)
    time.sleep(2)
    #html = each_page.page_source
    html = each_page.content
    soup = BeautifulSoup(html, 'html.parser')
    job_details = []
    try:
        #Extracting the posting title
        job_details.append('Title:' + soup.find('span', id="titletextonly").text)

        #Extracting all the compoenets as compensation, employment type etc.
        for span in soup.find_all('span', recursive=True):
            if not span.attrs.values():
                job_details.append(span.text)

        # adding city name
        city = link.strip()
        start = city.find("//") + len("//")
        end = city.find(".")
        substring = city[start:end]
        job_details.append('city:' + substring)

        # find post body
        post_body = soup.find(attrs={'id': 'postingbody'}).contents[2]
        # remove non ascii characters from post body
        job_details.append('post_body:' + re.sub("[^0-9a-zA-Z]+", " ", post_body))
        #Finding entire body or description to be precise
        for i in soup.find_all('section', {'id': 'postingbody'}):
            job_details.append('body:' + re.sub("[^0-9a-zA-Z]+", " ", (i.text.strip())))

        # appending postID
        job_details.append('pID:' + link.strip().replace('html', '').replace('.', '').split('/')[-1])


    except:
        pass
    # Dropping attributes without labels
    job_final = []
    for s in job_details:
        if ':' in s:
            job_final.append(s)

    # append clean attributes
    jobs.append(job_final)
    count += 1


#Function to store the list into a dictionary
def list_to_dict(rlist):
    return dict(s.split(':',1) for s in rlist)


# create a dictionary for label:value for each car attribute
job_dicts = []
for job in jobs:
    job_dict = list_to_dict(job)
    job_dicts.append(job_dict)

dfs = pd.DataFrame()
for item in job_dicts:
    df = pd.DataFrame.from_dict(item,orient='index').transpose()
    dfs= pd.concat([dfs,df], axis=0, ignore_index=True, sort=True)

dfs.to_csv('job.csv')