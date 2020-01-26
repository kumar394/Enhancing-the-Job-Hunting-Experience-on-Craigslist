import pandas as pd
import urllib
import bs4 as bs
from bs4 import SoupStrainer
from bs4 import BeautifulSoup


#Cites to Scrape information from
cities = ["indianapolis"]

jobs_id = []
jobs_title = []
jobs_url = []
jobs_description = []
jobs_category = []
jobs_location = []
jobs_compensation = []
jobs_employType = []

for city in cities:

    main_url = "https://" + city + ".craigslist.org/"
    html_main = BeautifulSoup(urllib.request.urlopen(main_url).read().decode('utf-8'), "html.parser")
    jobs = html_main.find('div', class_='jobs').find('div', class_='col')

    urls = []
    categories = []
    for li in jobs.find_all('a'):
        urls.append("https://" + city + ".craigslist.org" + li.get('href'))
        # Get list of all the jobs Category available in Craiglist
        categories.append(li.get_text())

    i = 1
    # Extract Job from each Category
    for url in urls[1:]:
        try:
            html = BeautifulSoup(urllib.request.urlopen(url).read().decode('utf-8'), "html.parser")

            total_count = int(html.find('span', class_='totalcount').get_text())
            count = 0

            while count <= total_count:

                li_rows = html.find('ul', class_='rows').find_all('li', class_='result-row')

                for li in li_rows:
                    # Incremental Job ID
                    jobs_id.append(len(jobs_id))

                    # Append Category and Location
                    jobs_category.append(categories[i])
                    jobs_location.append(city)
                    # Get Title of The job
                    jobs_title.append(li.find('a', class_="result-title hdrlnk").get_text())

                    # Get link of the Job Posting
                    desc_url = li.find('a', class_="result-title hdrlnk").get('href')
                    jobs_url.append(desc_url)

                    desc_html = BeautifulSoup(urllib.request.urlopen(desc_url).read().decode('utf-8'), "html.parser")

                    if desc_html.find('p', class_='attrgroup'):

                        compensation_done = False
                        type_done = False

                        # Get Compensation Type and Employment Type of the Job
                        side_bar = desc_html.find('p', class_='attrgroup')
                        span_all = side_bar.find_all('span')
                        for span in span_all:
                            if span.get_text()[0:12] == 'compensation':
                                compensation_done = True
                                jobs_compensation.append(span.get_text()[14:])
                            elif span.get_text()[0:10] == 'employment':
                                type_done = True
                                jobs_employType.append(span.get_text()[16:])

                        if not compensation_done:
                            jobs_compensation.append("NOT MENTIONED")
                        if not type_done:
                            jobs_employType.append("NOT MENTIONED")

                    else:
                        jobs_compensation.append("NOT MENTIONED")
                        jobs_employType.append("NOT MENTIONED")

                    # Get Desciption of the Job
                    body = desc_html.find('section', id="postingbody").get_text()
                    body = body.replace("QR Code Link to This Post", "").replace("\n", " ").replace("\t", " ")
                    jobs_description.append(body)

                count += 120
                if count <= total_count:
                    url_sub = url + "?s=" + str(count)
                    # Go to next page of the same category
                    html = BeautifulSoup(urllib.request.urlopen(url_sub).read().decode('utf-8'), "html.parser")
            i += 1
        except:
            # For the code to continue if there is error in a page
            i += 1
            print("Problem Occured at,  " + url)


#Create Pandas Dataframe of the extracted Job information
outputdf = pd.DataFrame(list(zip(jobs_id, jobs_category, jobs_location, jobs_title, jobs_compensation, jobs_employType, jobs_url, jobs_description)),
               columns =['Id', 'Category', 'Location', 'Title', 'Compensation', 'Type', 'Url', 'Description'])


#Export to CSV
outputdf.to_csv('Jobs_Final.csv', index=False)