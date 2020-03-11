import threading
import random
import time
import logging

def download_file(link, n, timeout=4):
    '''
    Pretend we are downloading data from a given *link*
    '''
    logging.info("Thread %d:   start downloading from %s", n, link)
    net_delay = random.randint(1, timeout+2) # simulate random network delay
    time.sleep(float(min(timeout, net_delay))) # wait for response before time out
    if (net_delay <= timeout):
        logging.info("Thread %d:   download successful! (%d seconds)", n, net_delay)
    else:
        logging.info("Thread %d:   download failed. (Timeout, %d seconds)", n, net_delay)


class DownloadThread(threading.Thread):

    def __init__(self, link, n, timeout):
        super().__init__()
        self.link = link
        self.n = n
        self.timeout = timeout

    def run(self):
        '''
        Pretend we are downloading data from a given *link*
        '''
        logging.info("Thread %d:   start downloading from %s", self.n, self.link)
        net_delay = random.randint(1, self.timeout+2) # simulate random network delay
        time.sleep(float(min(self.timeout, net_delay))) # wait for response before time out
        if (net_delay <= self.timeout):
            logging.info("Thread %d:   download successful! (%d seconds)", self.n, net_delay)
        else:
            logging.info("Thread %d:   download failed. (Timeout, %d seconds)", self.n, net_delay)


def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    # say we have three links
    link_pool = ["<link 1>", "<link 2>", "<link 3>"]
    thread_pool = list()
    network_timeout = 4
    
    for index, link in enumerate(link_pool):
        t = threading.Thread(target=download_file, args=(link, index, network_timeout))
        # also you can pass arguments with *kwargs*, which looks like
        # t = threading.Thread(target=download_file,\
        #     kwargs={"link":link, "n":index, "timeout":network_timeout})
        t.start() # start this thread, invoking target function, i.e. download_file()
        thread_pool.append(t)

    ''' # un-comment this part to create threads as DownloadThread objects
    for index, link in enumerate(link_pool):
        t = DownloadThread(link, index, network_timeout)
        t.start()
        thread_pool.append(t)
    '''

    for index, thread in enumerate(thread_pool):
        t.join() # wait for this thread before moving on

if __name__ == "__main__":
    main()