# 關閉 Rsoft 相關程序
# 當 Rsoft 跑模擬發生 SERVER 無法連上時執行
# 完畢後重啟 Rsoft

import os
os.system("taskkill /f /im rsssmpichmpd.exe")
os.system("taskkill /f /im hydra_pmi_proxy.exe")
os.system("taskkill /f /im dfmod.exe")