import os
import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time
#ヘッドレス実行　※全画面用
options = webdriver.ChromeOptions()
options.add_argument('--headless')


# chromedriver.exeがある場所
driver_path = "chromedriver-win64/chromedriver.exe"

#スクショ保存ディレクトリが存在しなければ生成
if os.path.isdir(os.getcwd()+"\\Chessboard_Images") == False:
    os.mkdir(os.getcwd()+"\\Chessboard_Images")

# webdriverの作成
service = Service(executable_path=driver_path) # 2) executable_pathを指定
driver = webdriver.Chrome(service=service) # 3) serviceを渡す

#URLに接続
driver.get("https://www.oculus.com/casting")
#ウインドウサイズをWebサイトに合わせて変更　※全画面用
width = driver.execute_script("return document.body.scrollWidth;")
height = driver.execute_script("return document.body.scrollHeight;")
driver.set_window_size(width,height)

#スクショをPNG形式で保存
i=0
while(1):
    #driver.get_screenshot_as_file(os.getcwd() + "\\Chessboard_Images\\" + fname + ".png")
    screenshot_name = "Chessboard_Images/"+f"screenshot_{i+1}.png"
    driver.save_screenshot(screenshot_name)
    i+=1
    if i>=99:
        i=0
    #print(f"Screenshot {i+1} taken: {screenshot_name}")
    