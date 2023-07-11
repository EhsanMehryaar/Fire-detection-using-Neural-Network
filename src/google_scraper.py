import os
import uuid
import time
import random
import urllib
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

def send_query_to_google_images(query, browser):
    """
    Send the search query to the Google Images search input field.
    
    :param query: str, search query
    :param browser: webdriver.Chrome object
    """
    search_input = browser.find_element_by_xpath('//input[@name="q"]')
    search_input.clear()
    search_input.send_keys(query)
    time.sleep(1)
    search_input.send_keys(Keys.ENTER)
    time.sleep(1)

def download_images_from_google(save_directory, num_images, browser):
    """
    Download the specified number of images from the Google Images search results.
    
    :param save_directory: str, directory to save the downloaded images
    :param num_images: int, number of images to download
    :param browser: webdriver.Chrome object
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    for i in range(num_images):
        try:
            image_link = browser.find_element_by_xpath(f'//div[@id="islrg"]//div[@data-ri="{i}"]//img')
            src_link = image_link.get_attribute('src')
            print(src_link)
            
            # Save the image using urllib
            img_name = uuid.uuid4()
            urllib.request.urlretrieve(src_link, os.path.join(save_directory, f'{img_name}.jpg'))
        except Exception as e:
            print(f"Error downloading image {i + 1}: {str(e)}")

def scrape_google_images(queries, save_root, num_images=500, is_headless=True):
    """
    Scrape Google Images for the specified queries and download images.
    
    :param queries: list of str, search queries
    :param save_root: str, root directory to save the downloaded images
    :param num_images: int, number of images to download per query
    :param is_headless: bool, whether to run the browser in headless mode (without GUI)
    """
    options = webdriver.ChromeOptions()
    if is_headless:
        options.add_argument('--headless')
    
    browser = webdriver.Chrome(options=options)
    browser.maximize_window()
    browser.get('https://www.google.com/imghp')
    time.sleep(random.random())
    
    for query in queries:
        save_directory = os.path.join(save_root, query.replace(" ", "_"))
        
        send_query_to_google_images(query, browser)
        download_images_from_google(save_directory, num_images, browser)
        
    browser.quit()

if __name__ == "__main__":
    queries = ["Forest fire", "Road fire", "Building fire", "Mountain fire", "Grassland fire", "Residential fire"]
    directory = r'F:\datasets\fire_images'
    num_images = 500

    scrape_google_images(queries, directory, num_images)
