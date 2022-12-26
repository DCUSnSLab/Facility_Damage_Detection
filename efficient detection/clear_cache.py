import os

import time



while(True):
    os.system('sysctl -w vm.drop_caches=3')
    time.sleep(3600)