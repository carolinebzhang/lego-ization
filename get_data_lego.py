import os
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Define the URL of the webpage
url = 'https://www.lego.com/en-us/themes/city'

options = Options()
options.headless = True
driver = webdriver.Chrome(options=options)
driver.get(url)

# Wait for all the images to load
wait = WebDriverWait(driver, 10)
wait.until(EC.presence_of_all_elements_located((By.TAG_NAME, 'img')))

soup = BeautifulSoup(driver.page_source, 'html.parser')

# Find all the images on the page
images = soup.find_all('div', class_='ProductImage_imageWrapper')


for i, image in enumerate(images):
    try: 
        image_url = image['src']
        if image_url.startswith('//'):
            image_url = f'https:{image_url}'
        elif image_url.startswith('/'):
            image_url = f'https://www.lego.com{image_url}'
        image_name = f'lego_set_{i}.jpg'
        image_path = os.path.join('lego_images', image_name)
        with open(image_path, 'wb') as f:
                f.write(requests.get(image_url).content)
                print(f'Saved {image_name} to {image_path}')
    except Exception as e:
        print(e)
        continue

driver.quit()
