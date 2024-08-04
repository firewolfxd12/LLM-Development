from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from time import sleep

def GetCleanedContent(text):
    x = 0
    while(x < len(text)):
        if(text[x] == '<'):
            index = text.find('>', x)
            if(text[x+1:index] == 'code'):
                x = text.find('</code>', x) + 7
            else:
                text = text.replace(text[x:index+1], "", 1)
        else:
            x += 1
    return text.replace('&nbsp;', ' ').strip(' ')

#Constansts
WaitTime = 3

#Selenium
options = Options()
# options.add_argument('--headless')
driver = webdriver.Firefox(service=Service(), options=options)

#File
f = open('data.txt', 'w')


for i in range(1, 66):
    #Get url with current page
    url = 'https://leetcode.com/problemset/?page='
    url += str(i)
    driver.get(url)

    sleep(WaitTime)

    problem_elements = driver.find_elements(By.CSS_SELECTOR, '.odd\\:bg-layer-1')

    #Create pairs (problem_name - link)
    links = []
    for elem in problem_elements:
        name_element = elem.find_element(By.CSS_SELECTOR, '.h-5')
        links.append((name_element.text, name_element.get_attribute('href')))
    
    #Iterate through all problems and get desciption
    for name, link in links:
        driver.get(link)
        sleep(WaitTime)
        
        description = driver.find_element(By.CLASS_NAME, "flexlayout__tab").find_element(By.CSS_SELECTOR, '[data-track-load="description_content"]')
        
        f.write(name + ':\n')
        for p in description.find_elements(By.TAG_NAME, "p"):
            if('example' in p.text.lower()):
                break
            text = p.get_attribute('innerHTML')
            f.write(GetCleanedContent(text))
        f.write('\n')

f.close()
driver.quit()
