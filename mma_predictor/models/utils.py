import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

def scrape_ufc_fighter_profile(driver, fighter_url):
    driver.get(fighter_url)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, 'c-hero__headline-prefix'))
    )
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    fighter_info = {
        'name': soup.find('span', class_='c-hero__headline-prefix').text.strip() if soup.find('span', class_='c-hero__headline-prefix') else 'N/A',
        'nickname': soup.find('p', class_='hero-profile__nickname').text.strip() if soup.find('p', 'hero-profile__nickname') else 'N/A',
        'height': soup.find('div', class_='field field--name-height').find('div', class_='field__item').text.strip() if soup.find('div', class_='field field--name-height') else 'N/A',
        'weight': soup.find('div', class_='field field--name-weight').find('div', class_='field__item').text.strip() if soup.find('div', class_='field field--name-weight') else 'N/A',
        'reach': soup.find('div', class_='field field--name-reach').find('div', class_='field__item').text.strip() if soup.find('div', class_='field field--name-reach') else 'N/A',
        'record': soup.find('div', class_='c-hero__headline-suffix tz-change-data').text.strip() if soup.find('div', class_='c-hero__headline-suffix tz-change-data') else 'N/A',
    }
    
    return fighter_info

def get_ufc_fighter_urls(driver, directory_url, num_fighters=50):
    driver.get(directory_url)
    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CLASS_NAME, 'c-listing-athlete-flipcard'))
    )
    
    fighter_urls = []
    while len(fighter_urls) < num_fighters:
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('/athlete/'):
                full_url = 'https://www.ufc.com' + href
                if full_url not in fighter_urls:
                    fighter_urls.append(full_url)
        
        if len(fighter_urls) >= num_fighters:
            break
        
        try:
            load_more_button = driver.find_element(By.CLASS_NAME, 'view-more')
            load_more_button.click()
            time.sleep(2)  # Wait for the next set of fighters to load
        except:
            break
    
    print(f"Scraped {len(fighter_urls)} fighter URLs.")
    return fighter_urls[:num_fighters]

def save_fighter_data(driver, fighter_urls, filepath):
    fighters = []
    for url in fighter_urls:
        fighter_info = scrape_ufc_fighter_profile(driver, url)
        print(f"Scraped data for fighter: {fighter_info['name']}")
        fighters.append(fighter_info)
    
    fighters_df = pd.DataFrame(fighters)
    print(f"Dataframe head: {fighters_df.head()}")
    fighters_df.to_csv(filepath, index=False)
    return fighters_df
