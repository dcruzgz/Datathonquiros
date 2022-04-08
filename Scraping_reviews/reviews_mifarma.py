from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import io
import pandas as pd

# add your chrome driver path here
url = 'https://www.google.com/maps/place/Mifarma+by+Atida+Plus/@38.9921697,-1.8554824,15z/data=!4m7!3m6!1s0x0:0xd53c2bccdd11b7b9!8m2!3d38.9921697!4d-1.8554824!9m1!1b1'
path = 'C://Users//dcruzg//Desktop//Datathon//Atmira_Pharma_Visualization//dathon//scraping'
chromedrive_path = path + './chromedriver'
browser = webdriver.Chrome(chromedrive_path)
actions = ActionChains(browser)
# add your google map link whose data you want to scrape
browser.get(url)
browser.maximize_window()
time.sleep(2)
pas = browser.find_element_by_xpath('//*[@id="yDmH0d"]/c-wiz/div/div/div/div[2]/div[1]/div[4]/form/div/div/button').click()
time.sleep(3)
content = browser.find_element_by_xpath('//*[@id="pane"]/div/div[1]/div/div/div[2]/div[5]').click() ##Yr7JMd-pane-content cYB2Ge-oHo7ed
htmlstring = browser.page_source
afterstring = ""
for i in range(12):
    afterstring = htmlstring
    actions.send_keys(Keys.PAGE_DOWN).perform()
    htmlstring = browser.page_source + htmlstring
    if (i > 12):
        print("ended scraping crack test one")
        actions.send_keys(Keys.PAGE_DOWN).perform()
        htmlstring = browser.page_source + htmlstring
        if (i > 12):
            print("--Scrapping End--")
            break
    time.sleep(3)

textdoc = io.open("data.txt", "a+", encoding="utf-8")
soup = BeautifulSoup(htmlstring, "html.parser")
mydivs = soup.findAll("div", {"class": "ODSEW-ShBeI NIyLF-haAclf fontBodyMedium"})

counter = 0
Reviwer_data = {'Index': [], 'Reviewer Name': [], 'Reviewer Rating': [], 'Reviewer Profile URL': [], 'Review': [], 'Time': []}

for a in mydivs:
    if ('Index', counter) in Reviwer_data.items():
        continue
    else:
        textdoc.write(str(
            "\nReviewer name: " + a.find("div",
                                         class_="ODSEW-ShBeI-title").text) + " \n||Reviewerer Profile URL:" + str(
            a.find("a").get('href')))
        textdoc.write(" \n||Review:" + a.find("span", class_="ODSEW-ShBeI-text").text + " \n||Time: " + a.find("span",
                                                                                                               class_="ODSEW-ShBeI-RgZmSc-date").text)
        textdoc.write("\n")
        textdoc.write(str(a.find("span", class_="ODSEW-ShBeI-H1e3jb")))
        textdoc.write("=========================================\n")
        Reviwer_data['Index'].append(counter)
        Reviwer_data['Reviewer Name'].append(str(a.find("div", class_="ODSEW-ShBeI-title").text).strip())
        Reviwer_data['Reviewer Rating'].append(str(a.find("span", class_="ODSEW-ShBeI-H1e3jb")))
        Reviwer_data['Reviewer Profile URL'].append(str(a.find("a").get('href')))
        Reviwer_data['Review'].append(a.find("span", class_="ODSEW-ShBeI-text").text)
        Reviwer_data['Time'].append(a.find("span", class_="ODSEW-ShBeI-RgZmSc-date").text)
        counter = counter + 1
print("Total reviews scraped:" + str(counter))
textdoc.close()
pd.DataFrame(Reviwer_data).to_csv('data.csv',sep=';', index= False)



