from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import os

webdriver_path = r"P:\碩二\二下\高等人工智慧\期末報告\chromedriver.exe"

# 設定 Chrome driver（可選擇無頭模式）
options = Options()
options.add_argument('--headless')  # 如果您希望瀏覽器在背景執行
service = Service(executable_path=webdriver_path)  # 請替換為您的 chromedriver 路徑
driver = webdriver.Chrome(service=service, options=options)

try:
    # 開啟目標網頁
    url = "https://data.taipei/dataset/detail?id=660e3969-c011-481c-aa24-8a949ac2d62d"
    driver.get(url)

    # 等待「顯示全部」按鈕可被點擊，然後點擊它
    try:
        show_all_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//span[text()='顯示全部']/.."))
        )
        show_all_button.click()
        time.sleep(3)  # 等待資料載入
    except Exception as e:
        print("找不到「顯示全部」按鈕或載入失敗", e)

    # 取得完整的 HTML
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    # 擷取所有下載連結
    links = []
    for a in soup.select('a.download-file-btn'):
        href = a.get('href')
        if href:
            links.append(href)

    # 輸出所有連結
    for link in links:
        print(link)

finally:
    driver.quit()
