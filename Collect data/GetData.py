from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from time import sleep
import markdownify
import json

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

def GetDescription(driver, name):
    description = driver.find_element(By.CLASS_NAME, "flexlayout__tab").find_element(By.CSS_SELECTOR, '[data-track-load="description_content"]')
    
    global data

    for p in description.find_elements(By.TAG_NAME, "p"):
        if('example' in p.text.lower()):
            break
        text = GetCleanedContent(p.get_attribute('innerHTML'))
        text = markdownify.markdownify(text, heading_style='SETEXT')
        data[name]["problem"] = data[name].get("problem", "") + text

    data[name]["problem"] += '\n'

def GetSolution(driver, name):
    solution = driver.find_element(By.CLASS_NAME, "FN9Jv")
    approaches = solution.find_elements(By.TAG_NAME, "h3")
    a = solution.get_attribute('innerHTML')
    a = a[a.find(approaches[-1].text):]
    
    global data
    
    m = markdownify.markdownify(a[a.find('Algorithm'):a.find('Implementation')], heading_style = 'SETEXT')
    data[name]["explanation"] = m

    urls = solution.find_elements(By.TAG_NAME, "iframe")
    implementation_url = ''
    
    for i in range(len(urls)-1, -1, -1):
        if('https://leetcode.com/playground/' in urls[i].get_attribute('src')):
            implementation_url = urls[i].get_attribute('src')
            break
    try:
        driver.get(implementation_url)
        sleep(WaitTime)

        buttons = driver.find_element(By.CLASS_NAME, "lang-btn-set").find_elements(By.TAG_NAME, "button")

        for button in buttons:
            if('py' in button.text.lower()):
                button.click()

        sleep(1)

        lines = driver.find_element(By.CLASS_NAME, "CodeMirror-lines").find_element(By.CLASS_NAME, "CodeMirror-code").find_elements(By.TAG_NAME, "pre")
        
        code = '```python\n'
        for line in lines:
            code += line.text + '\n'
        code += '```'
        
        data[name]["solution"] = code
    except Exception as e:
        print(repr(e))
    
#Constansts
WaitTime = 3

#Selenium
options = Options()
options.add_argument('--headless')
driver = webdriver.Firefox(service=Service(), options=options)

#File and data
f = open('data.json', 'w')
data = dict()

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
        link = name_element.get_attribute('href')
        if('?' in link):
            link = link[:link.find('?')]
        links.append((name_element.text, link))
    
    #Iterate through all problems and get desciption
    for name, link in links:
        data[name] = {}
        driver.get(link)
        sleep(WaitTime)
        GetDescription(driver, name)
        new_link = ''
        if(link[-1] == '/'):
            new_link = link + 'editorial'
        else:
            new_link = link + '/editorial'
        driver.get(new_link)
        sleep(WaitTime)
        try:
            GetSolution(driver, name)
        except Exception as e:
            print(repr(e))
f.write(json.dumps(data, indent=4, ensure_ascii=False))
f.close()
driver.quit()

