import threading
import logging
import time
import argparse

class Database:

    def __init__(self, use_lock=True):
        self.val = 0
        self._lock = threading.Lock()
        self.use_lock = use_lock

    def increment_one(self):
        if self.use_lock:
            self._lock.acquire()

        # simulate a read-modify-write process
        local_copy = self.val
        logging.info("get value %d, try to increment by one", self.val)
        local_copy += 1
        time.sleep(0.5) # due to processing or network delay
        self.val = local_copy
        logging.info("write value %d to database", local_copy)
        
        if self.use_lock:
            self._lock.release()

def worker(database):
    database.increment_one()

def main():
    # python testLock.py 0 : don't use lock, race condition happens
    # python testLock.py 1 : use lock, race condition is avoided
    parser = argparse.ArgumentParser(description="a simple example of using threading.Lock to avoid race conditions")
    parser.add_argument("use_lock", type=int, help="tell me to use lock (1) or not (0)?")
    args = parser.parse_args()

    logging.basicConfig(format="%(message)s", level=logging.INFO)

    database = Database(args.use_lock)

    t1 = threading.Thread(target=worker, args=(database, ))
    t2 = threading.Thread(target=worker, args=(database, ))

    logging.info("(before) val: %d, starting threads...\n", database.val)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    logging.info("\n(after) val: %d", database.val)

if __name__ == "__main__":
    main()