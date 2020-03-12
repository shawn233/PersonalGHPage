from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=5)

import time
def wait_a_while(seconds, index):
    '''
    say we want a thread to wait for a while
    '''
    print("[future {}] sleeping for {} seconds ...".format(index, seconds))
    time.sleep(seconds)
    print("[future {}] waken up after {} seconds".format(index, seconds))
    return seconds

# provision tasks
future_1 = executor.submit(wait_a_while, 5.0, 1)
future_2 = executor.submit(wait_a_while, 2.0, 2)

# wait for returns
print("[  main  ] future 1 returns {}".format(future_1.result()))
print("[  main  ] future 2 returns {}".format(future_2.result()))

# free resources after current pending futures complete
executor.shutdown(wait=True)
print("all done!")