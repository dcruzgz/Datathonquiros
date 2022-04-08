from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import io
import pandas as pd
import lxml
import cchardet

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

textdoc = io.open("data.txt", "a+", encoding="utf-8")

Reviwer_data = {'Index': [], 'Reviewer Name': [], 'Reviewer Rating': [], 'Review': [], 'Time': []}

counter = 0
for i in range(1000):
    print(i)
    afterstring = htmlstring
    actions.send_keys(Keys.PAGE_DOWN).perform()
    htmlstring = browser.page_source
    textdoc.write(htmlstring)
    soup = BeautifulSoup(htmlstring, 'lxml')
    mydivs = soup.findAll("div", {"class": "ODSEW-ShBeI NIyLF-haAclf fontBodyMedium"})
    for a in mydivs:
        list = Reviwer_data['Reviewer Name']
        act_name = str(a.find("div", class_="ODSEW-ShBeI-title").text).strip()
        if act_name in list:
            continue
        else:
            Reviwer_data['Index'].append(counter)
            Reviwer_data['Reviewer Name'].append(str(a.find("div", class_="ODSEW-ShBeI-title").text).strip())
            Reviwer_data['Reviewer Rating'].append(str(a.find("span", class_="ODSEW-ShBeI-H1e3jb")))
            Reviwer_data['Review'].append(a.find("span", class_="ODSEW-ShBeI-text").text)
            Reviwer_data['Time'].append(a.find("span", class_="ODSEW-ShBeI-RgZmSc-date").text)
            counter = counter + 1

    if (i > 1000):
        print("ended scraping crack test one")
        actions.send_keys(Keys.PAGE_DOWN).perform()
        htmlstring = browser.page_source
        textdoc.write(htmlstring)
        if (i > 1000):
            print("--Scrapping End--")
            break
    time.sleep(3)

textdoc.close()
pd.DataFrame(Reviwer_data).to_csv('data.csv',sep=';', index= False)
browser.quit()


