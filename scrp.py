from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import pandas as pd

service = Service(executable_path='chromedriver.exe')
driver = Chrome()

wait = WebDriverWait(driver, 10)
url = 'https://uspdigital.usp.br/jupiterweb/listarGradeCurricular?codcg=81&codcur=81102&codhab=2&tipo=N'

driver.get(url)
wait.until(
    EC.presence_of_element_located(
        (By.XPATH, '(//*[text()[contains(.,"Eletivas")]])[1]')
    )
)
trs = driver.find_elements(By.XPATH, '(//*[text()[contains(.,"Eletivas")]])[1]/preceding::tr[@bgcolor="#FFFFFF" and @class="txt_verdana_8pt_gray"]')
codes = [tr.find_elements(By.XPATH, 'td')[0].text for tr in trs]
names = [tr.find_elements(By.XPATH, 'td')[1].text for tr in trs]

df = pd.DataFrame(
    {
        'code': codes,
        'name': names
    }
)

df.to_csv('eco.csv', index=False)