from selenium import webdriver
from time import sleep
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException, ElementClickInterceptedException
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support import expected_conditions as EC


options = webdriver.ChromeOptions()
options.add_argument("--disable-extensions")
options.add_argument("--disable-gpu")
options.add_argument("--incognito")
options.add_argument("--lang=en")
options.add_argument("--headless")
options.add_argument("--disable-webgl")
options.add_argument('--log-level=3')

ser = Service('../chromedriver_win64/chromedriver.exe')
driver = webdriver.Chrome(service=ser, options=options)

wait = WebDriverWait(driver, 10)

driver.get("https://www.google.fr/")

if driver.find_elements(By.XPATH, '//*[@id="L2AGLb"]'):
    print("autoriser les cookies")
    button_cookies = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="L2AGLb"]')))
    button_cookies.click()
    print("click")

if driver.find_elements(By.XPATH, '/html/body/div[5]/div[1]/div/div[2]/div/div/div/div/div[2]/div/button[1]'):
    print("autoriser les cookies")
    button = wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/div[1]/div/div[2]/div/div/div/div/div[2]/div/button[1]')))
    button.click()
    print("click")


            
driver.close()

#By.CLASS_NAME, "x1iyjqo2"
