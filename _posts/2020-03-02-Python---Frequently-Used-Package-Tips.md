---
layout:     post
title:      Python | Frequently Used Package Tips
subtitle:   personal cheatsheet for frequently used Python packages
date:       2020-03-02
author:     Xinyu Wang
header-img: img/post-bg-cook.jpg
catalog:    true
tags:
    - Python
---

<!--
<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>
-->



**NOTICE**: I'm having some formatting problem with this post. Sample codes look like a mess on this page. Click [here](https://github.com/shawn233/shawn233.github.io/blob/master/_posts/2020-03-02-Python---Frequently-Used-Package-Tips.md) (Github source file) to have a better reading experience.

As a dedicated Python supporter, I use Python to handle miscellaneous programming tasks every day, from writing a simple scipt to modify file extension (e.g., .txt -> .md) in batches, to running neural network models.

A common dilemma for crazy Python programmers like me, is to remember how to use uncountable packages. For one task, I learned to use numpy, matplotlib, and scipy; for another task, I turned to the docs of pandas, pillow and pytorch.  But when I get back to matplotlib, I have to read the docs again because when I stop using a package, I tend to forget quickly. 

Now you see why I intend to write this post. In this post, I'll note down the core functions of frequently used packages, storing my knowledge about a package in the most compact way while I still can.

This post is helpful to me, bringing up some of my faded memories. If you are interested, go ahead and read it! You'll surely find the part that works for you.

# OS.Path | Common pathname manipulations 

[os.path module](https://docs.python.org/3/library/os.path.html#module-os.path) is a ferequently used module to manipulate file paths. File paths are encouraged to represent as string objects.

Here are some frequently used methods, with explanation and samples.

```python
# interactive environment
>>> root = os.getcwd()  # get current working directory
>>> print("the root directory is", root)
the root directory is /home/shawn233
```

[*abspath()* method](https://docs.python.org/3/library/os.path.html#os.path.abspath) appends the path name to the current working directory

```python
>>> # 
>>> os.path.abspath("foo")
'/home/shawn233/foo'
>>> os.path.abspath("foo/bar")
'/home/shawn233/foo/bar'
```

[*split()* method](https://docs.python.org/3/library/os.path.html#os.path.split) divides the path name into a pair, (*dir*, *base*). If path ends in a slash, *base* will be empty. If there is no slash in path, *dir* will be empty.

You can also call [*dirname()*](https://docs.python.org/3/library/os.path.html#os.path.dirname) or [*basename()*](https://docs.python.org/3/library/os.path.html#os.path.basename) method to get the first or second element of the pair returned by *split()*.

```python
>>> # .
>>> os.path.split("/foo/bar")
('/foo', 'bar')
>>> os.path.split("/foo/bar/")
('/foo/bar', '')
>>> os.path.split("foo")
('', 'foo')
>>> os.path.split("")
('', '')
>>>
>>> # basename returns the base name of the path name
>>> os.path.basename("/foo/bar")
'bar'
>>> os.path.basename("/foo/bar/")
''
>>>
>>> # dirname returns the directory name of the path name
>>> os.path.dirname("/foo/bar")
'/foo'
>>> os.path.dirname("/foo/bar/")
'/foo/bar'
```

[*splitext()* method](https://docs.python.org/3/library/os.path.html#os.path.splitext) splits the path name into a pair (*root*, *ext*). *ext* either is empty or begins with a period (.), and contains at most one period. Note that the leading periods are not considered to put in *ext*.

```python
>>> os.path.splitext("/foo/bar.py")
('/foo/bar', '.py')
>>> os.path.splitext("/foo/bar.ext.py")
('/foo/bar.ext', '.py')
>>> os.path.splitext(".bar")
('.bar', '')
>>> os.path.splitext("...bar")
('...bar', '')
```

[*exists()* method](https://docs.python.org/3/library/os.path.html#os.path.exists) returns True if path name refers to an existing path or an open file descriptor. Returns False for broken symbolic links.

```python
>>> os.path.exists("/home/shawn233/Documents/")
True
>>> os.path.exists("/home/shawn233/Documents")
True
>>> os.path.exists("/home/shawn233/foo")
False
```

[*join()* method](https://docs.python.org/3/library/os.path.html#os.path.join) joins one or more path components intelligently. If a component is an absolute path, all previous components are thrown away and joining continues from the absolute path component.

```python
>>> os.path.join("foo", "bar", "file.py")
'foo/bar/file.py'
>>> os.path.join("/foo", "bar", "file.py")
'/foo/bar/file.py'
>>> os.path.join("/foo", "/bar", "file.py")
'/bar/file.py'
```

[*expanduser()* method](https://docs.python.org/3/library/os.path.html#os.path.expanduser) expands an initial path component `~` or `~user` in the given *path* to *user*'s home directory. On Unix platform, the underlying mechanism is to replace `~` with the value of *HOME* environment variable (access by `os.environ["HOME"]`). When the initial component is `~user`, only a valid user name will be expanded.

```python
>>> path = "./data/facescrub"
>>> os.path.expanduser(path)
'./data/facescrub'
>>> path = "~/data/facescrub"
>>> os.path.expanduser(path)
'/home/shawn233/data/facescrub'
>>> path = "~alice/data/facescrub" # no alice on my computer
>>> os.path.expanduser(path)
'~alice/data/facescrub'
>>> path = "~shawn233/data/facescrub" # shawn233 is a valid user
>>> os.path.expanduser(path)
'/home/shawn233/data/facescrub' 
```

Besides the mentioned functionalities, [os.path module](https://docs.python.org/3/library/os.path.html#module-os.path) can also be used to get path information (*getatime()*, *getmtime()*, *getctime()*, etc.), and test path types (*isabs()*, *isfile()*, *isdir()*, *islink()*, *ismount()*, *samefile()*, *sameopenfile()*, *samestat()*, etc.). Explore more methods of os.path in the official docs.

# OS | Miscellaneous operating system interfaces

Here are a few frequently used methods that support traversing directories.

- [*os.walk()*](https://docs.python.org/3/library/os.html#os.walk): display a directory tree, expanding all subdirectories, recursively
- [*os.scandir()*](https://docs.python.org/3/library/os.html#os.scandir): iterate over all entries (files and subdirectories) in a given directory
- [*os.listdir()*](https://docs.python.org/3/library/os.html#os.listdir): return a list containing names of all entries in a given directory

Let's say we have a folder that stores all of our studying materials, named `lectures`, in which we have the following folders and files.

```bash
$ tree lectures/
lectures/
├── English
│   ├── listening
│   │   └── listen.mp3
│   ├── reading
│   │   └── read.pdf
│   ├── speaking
│   │   └── speak.mkv
│   └── writing
│       └── write.docx
├── Maths
│   └── slides.pptx
├── Programming
│   └── main.cpp
└── syllabus.txt

7 directories, 7 files
```

Now we access the `lectures` folder, with three different *os* methods.

The first method is *os.listdir()*. It returns a list of all entry names in the first layer of the directory tree.

```python
>>> import os
>>> entries = os.listdir("./lectures/")
>>> entries
['Programming', 'syllabus.txt', 'Maths', 'English']
```

The second method is *os.scandir()*. It returns an iterator of [*os.DirEntry*](https://docs.python.org/3/library/os.html#os.DirEntry) objects corresponding to all entries in the first layer of the directory tree. So the results should be similar to *os.listdir()* to some extent. *stat* info can also be accessed by  *DirEntry* objects.

```python
>>> import os
>>> for entry in os.scandir("./lectures/"):
...     print(entry.name)
...     print("  path", entry.path)
...     print("  is_file", entry.is_file())
...     print("  is_dir", entry.is_dir())
... 
Programming
  path ./lectures/Programming
  is_file False
  is_dir True
syllabus.txt
  path ./lectures/syllabus.txt
  is_file True
  is_dir False
Maths
  path ./lectures/Maths
  is_file False
  is_dir True
English
  path ./lectures/English
  is_file False
  is_dir True
```

The third method is *os.walk()*. It generates the whole directory tree, expanding all subdirectories. It is likely a higher-level method which internally calls *os.scandir()*. By assigning the parameter *topdown* as `True`, this method runs a root-first traverse on the directory tree. The official docs say the *dirnames* can be changed in-place, e.g., *del* the *dirnames* list may prune the subsequent visits to subdirectories in this list.

```python
>>> import os
>>> for dirpath, dirnames, filenames in os.walk("./lectures/", topdown=True):
...     print(dirpath)
...     print("  subdirs", dirnames)
...     print("  files", filenames)
... 
./lectures/
  subdirs ['Programming', 'Maths', 'English']
  files ['syllabus.txt']
./lectures/Programming
  subdirs []
  files ['main.cpp']
./lectures/Maths
  subdirs []
  files ['slides.pptx']
./lectures/English
  subdirs ['listening', 'reading', 'speaking', 'writing']
  files []
./lectures/English/listening
  subdirs []
  files ['listen.mp3']
./lectures/English/reading
  subdirs []
  files ['read.pdf']
./lectures/English/speaking
  subdirs []
  files ['speak.mkv']
./lectures/English/writing
  subdirs []
  files ['write.docx']

```



# Fileinput | Iterate over lines from multiple input streams

[*fileinput* module](https://docs.python.org/3/library/fileinput.html) implements a helper class and functions to quickly write a loop over standard input or a list of files.

# Sys | System-specific parameters and functions

[*sys.argv*](https://docs.python.org/3/library/sys.html#sys.argv) is a list of command line arguments passed to a Python script.

For example, the following script prints all command line arguments in order.

```python
import sys
for i in range(len(sys.argv)):
    print("argv[{}]: {}".format(i, sys.argv[i]))
```

*sys.argv[0]* is always the script name. It runs like:

```bash
$ python foo.py
argv[0]: foo.py
$
$ python foo.py a b c d
argv[0]: foo.py
argv[1]: a
argv[2]: b
argv[3]: c
argv[4]: d
```

[*sys.stdin*](https://docs.python.org/3/library/sys.html#sys.stdin), [*sys.stdout*](https://docs.python.org/3/library/sys.html#sys.stdout), and [*sys.stderr*](https://docs.python.org/3/library/sys.html#sys.stderr) are text file objects used by the interpreter for standard input, output and errors.

- *sys.stdin* is used for all interactive input;
- *sys.stdout* is used for the output of *print()* and [expression](https://docs.python.org/3/glossary.html#term-expression) statements and for the prompts of *input()*;
- The interpreter's own prompts and its error messages go to *sys.stderr*.

# Numpy



[*concatenate()* method](https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html) joins a sequence of arrays along an existing axis.

```python
>>> a = np.ones((3, 4))
>>> a
array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]])
>>> b = np.zeros((3, 4))
>>> b
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])
>>>
>>> c0 = np.concatenate([a, b], axis=0)
>>> c0.shape
(6, 4) # concatenating axis=0, so shape[0] is accumulated
>>>
>>> c1 = np.concatenate([a, b], axis=1)
>>> c1.shape
(3, 8) # concatenating axis=1, so shape[1] is accumulated
>>>
>>> c0
array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])
>>> c1
array([[1., 1., 1., 1., 0., 0., 0., 0.],
       [1., 1., 1., 1., 0., 0., 0., 0.],
       [1., 1., 1., 1., 0., 0., 0., 0.]])
```

[*stack()* method](https://docs.scipy.org/doc/numpy/reference/generated/numpy.stack.html) joins a sequence of arrays along a new axis. The *axis* argument indicates where the new axis should be placed.

For a sequence of *x* arrays that are *(m, n)* shaped, the output shape for different values of *axis* is listed below.

- *axis=0*: *(x, m, n)*
- *axis=1*: *(m, x, n)*
- *axis=2* (also written as *axis=-1*): *(m, n, x)*

```python
>>> import numpy as np
>>>
>>> num1 = np.zeros(shape=(3, 4), dtype=np.int)
>>> num2 = np.ones(shape=(3, 4), dtype=np.int)
>>>
>>> num3_0 = np.stack([num1, num2], axis=0)
>>> print("shape:", num3_0.shape, "\n\n", num3_0)
shape: (2, 3, 4) 

 [[[0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]]

 [[1 1 1 1]
  [1 1 1 1]
  [1 1 1 1]]]
>>>
>>> num3_1 = np.stack([num1, num2], axis=1)
>>> print("shape:", num3_1.shape, "\n\n", num3_1)
shape: (3, 2, 4) 

 [[[0 0 0 0]
  [1 1 1 1]]

 [[0 0 0 0]
  [1 1 1 1]]

 [[0 0 0 0]
  [1 1 1 1]]]
>>>
>>> num3_2 = np.stack([num1, num2], axis=2)
>>> print("shape:", num3_2.shape, "\n\n", num3_2)
shape: (3, 4, 2) 

 [[[0 1]
  [0 1]
  [0 1]
  [0 1]]

 [[0 1]
  [0 1]
  [0 1]
  [0 1]]

 [[0 1]
  [0 1]
  [0 1]
  [0 1]]]
```

Although the examples above may look confusing, in most cases we only need to remember that if we want to create a new axis, we should use *stack()* instead of *concatenate()*. 



# Pandas



# Matplotlib



# PyTorch



# Argparse | Parse for command-line options, arguments and sub-commands

```python
import argparse
```

argparse is a frequently used package that supports convenient management of parameters passed from the command line. It supports several types of parameters, including:

- positional arguments: users are obligatory to provide these arguments to run a program 
- optional arguments: users may choose to provide these arguments out of their personal intentions

A program that uses argparse looks like this.

```bash
$ python foo.py --help
usage: foo.py [-h] [-n NUMBER] [-s SEPERATOR] word

a simple scipt to repeat words

positional arguments:
  word                  the word to repeat

optional arguments:
  -h, --help            show this help message and exit
  -n NUMBER, --number NUMBER
                        repeat the word for <number> times
  -s SEPERATOR, --sep SEPERATOR
                        seperators inserted between every word pairs
$
$ python foo.py harry
harry harry harry harry harry                  
$
$ python foo.py harry -n 2 -s \ is\ 
harry is harry
$
$ python foo.py O --number 3 --sep RERE
OREREOREREO
```

To start with defining parameters, create an [*ArgumentParser*](https://docs.python.org/3.7/library/argparse.html#argumentparser-objects) object. You may add a proper description to help users understand your program.

```python
parser = argparse.ArgumentParser(description="a simple scipt to repeat words")
```

Add positional and optional arguments with [*add_argument()*](https://docs.python.org/3.7/library/argparse.html#the-add-argument-method) method.

```python
parser.add_argument("word", help="the word to repeat", type=str)
parser.add_argument("-n", "--number", help="repeat the word for <number> times", dest="number", type=int, default=5)
parser.add_argument("-s", "--sep", help="seperators inserted between every word pairs", dest="seperator", type=str, default=" ")
args = parser.parse_args()
```

Call [*ArgumentParser.parse_args()*](https://docs.python.org/3.7/library/argparse.html#the-parse-args-method) and get a namespace comprised of arguments defined by the [*add_argument()*](https://docs.python.org/3.7/library/argparse.html#the-add-argument-method) method.

```python
args = parser.parse_args()
```

With the range of variable *args* returned by the *parse_args()* method, values of arguments are held in *args* as its attributes. You can access these values by accessing *args* attributes with the argument name as the attribute name. For example,

```python
# $ python foo.py harry -n 3 -s **
# parser = argparse.ArgumentParser()
# ... adding arguments
args = parser.parse_args()
print(args.word)        # harry
print(args.number)      # 3
print(args.seperator)   # **
```

# Threading | Thread-based parallelism

[threading module](https://docs.python.org/3/library/threading.html) allows you to implement **parallelism** in a convenient way.

```python
import threading
```

(Using the threading module usually includes a large amount codes, such as worker definition, thread creation, and main thread implementation. So due to restricted space, sample codes in this section are mostly provided as pseudocodes. But don't worry, these codes are fully commented and clear enough to get the idea. For more curious readers, I also uploaded some [realistic samples](https://github.com/shawn233/shawn233.github.io/tree/master/_posts/related-codes/threading-samples) that you can play with.)

[*threading.Thread*](https://docs.python.org/3/library/threading.html#threading.Thread) objects are created to actually work as threads. There are mainly two ways to designate the routine that the thread executes.

- construct a *Thread* object with a *target* argument
- override the *run()* method in a subclass

The *target* argument accepts a callable object, to be invoked by the *run()* method.

Let's say we would like to download some data from a set of given links. Due to terrible network, downloading a file takes as slow as several seconds. Since at most time our program is just waiting for response from remote servers, what we are writing is an IO-bounded program.

With the help of threading module, this program get boosted using multiple threads to request multiple servers at the same time.

```python
# pseudocode
link_pool = ["link 1", "link 2", ...]
thread_pool = list()

for link in link_pool:
    thread = threading.Thread(target=download, args=(link,))
    thread_pool.append(thread)

for thread in thread_pool:
    thread.join()
```

Overriding *\_\_init\_\_()* and *run()* methods also works.

```python
# pseudocode
class DownloadThread(threading.Thread):
    def __init__(self, link):
        super().__init__()
        self.link = link
    def run(self):
        # to be invoked by .start() method
        download_from(self.link)

link_pool = ["link 1", "link 2", ...]
thread_pool = list()
for link in link_pool:
    thread = DownloadThread(link)
    thread.start()

for thread in thread_pool:
    thread.join()
```

Besides *Thread*, threading module has an extensive set of useful classes to facilitate multi-thread programming.

[*threading.Lock*](https://docs.python.org/3/library/threading.html#threading.Lock) provides a basic primitive to avoid race conditions.

```python
class Database:
    def __init__(self):
        self.val = 0
        self._lock = threading.Lock()
    def increment_one(self):
        self._lock.acquire()
        # simulate read-modify-write
        local_copy = self.val() # get val
        local_copy += 1 # increment by 1
        time.sleep(0.5) # processing delay
        self.val = local_copy # database updated
        self._lock.release()
```

Although it is quite intuitive to use a lock object by invoking its *acquire()* and *release()* methods, it is strongly recommended to use the lock as a context manager (*with* statement), because a context manager ensures the lock is automatically released when the *with* block exits for any reason, especially due to exceptions.

```python
with self._lock:
    # do something ...
```

is equivalent to:

```python
self._lock.acquire()
try:
    # do something ...
finally:
    self._lock.release()
```

(Context managers also work for *threading.Condition* and *threading.Semaphore* objects.)

[*threading.Semaphore*](https://docs.python.org/3/library/threading.html#threading.Semaphore) implements semaphore, which manages an atomic (thread-safe) counter. Invoking *acquire()* method on the *Semaphore* object decrements the counter value by 1. When the counter value is 0, invoking *acquire()* method blocks the thread until another thread invokes *release()* method which increments counter value by 1.

```python
>>> import threading
>>> sema = threading.Semaphore(value=1) # counter set to 1
>>> sema.release() # unreasonable but allowed, counter:1->2
>>> sema.acquire(blocking=True, timeout=1) # counter:2->1
True
>>> sema.acquire(blocking=True, timeout=1) # counter:1->0
True
>>> sema.acquire(blocking=True, timeout=1) # counter:0
False
>>> sema.release() # counter: 0->1
>>> sema.acquire(blocking=True, timeout=1) # counter: 1->0
True
```

A normal *Semaphore* object can be released before any *acquire()* calls, which is unreasonable and even dangerous sometimes. Say we have a database server which supports a maximum connection of 5, if we invoke *release()* before any connection, now 6 connections may be established, and a server crash is potential to occur.

So, [*threading.BoundedSemaphore*](https://docs.python.org/3/library/threading.html) is more frequently used. It ensures the counter value doesn't exceed its initial value. If it does, *ValueError* is raised.

```python
# use BoundedSemaphore to protect a poor database server
maxconnections = 5
pool_sema = threading.BoundedSemaphore(value=maxconnections)

# in each thread, acquire the semaphore before connecting
def worker():
    with pool_sema:
        conn = connectdb()
        try:
            # use connection ...
        finally:
            conn.close()
```

[*threading.Event*](https://docs.python.org/3/library/threading.html#threading.Event) is one of the simplest mechanism for communication between threads: one thread signals an event and other threads wait for it. Internally, an *Event* object is a boolean flag that is initialized as *False*.

```python
# pseudocode
event = threading.Event()

def image_downloader(link, event):
    img = download_from(link)
    time.sleep(3.0) # simulate network delay
    event.set()
    
def image_processor(event):
    event.wait() # block until event is set
    img = read_local_image()
    process_image(img)
```

[*threading.Timer*](https://docs.python.org/3/library/threading.html#threading.Timer) invokes a given *function* after *interval* seconds have passed.

```python
def hello():
    print("hello timer")

t = Timer(5.0, hello)
t.start() # after 5 seconds, "hello timer will be printed"
```

[*threading.Barrier*](https://docs.python.org/3/library/threading.html#threading.Barrier) is a simple synchronization primitive for use by a fixed number threads that need to wait for each other.

About **Daemons**. Python waits for non-daemonic threads to complete before termination, while kills threads that are daemons when the program is exiting. Calling *Thread.join()* method on a daemon thread, however, makes the program waits for it to complete. ([source](https://realpython.com/intro-to-python-threading/#starting-a-thread)) see section *Daemon Threads*. (*threading.Thread* class has a boolean argument named *daemon* to set this attribute.) 

It is remarkable that despite that functions seem to run in parallel with the help of this module, internally they still run in a single-threaded way rather than a concurrent way, due to the limit of [Python Global Interpreter Lock (GIL)](https://realpython.com/python-gil/), which only allows one thread to acquire the interpreter lock at a time. This feature may cause trouble to CPU-bound multi-threaded programs. Here is what the official doc says about this feature:

> If you want your application to make better use of the computational resources of multi-core machines, you are advised to use [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing) or [`concurrent.futures.ProcessPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor). However, threading is still an appropriate model if you want to run multiple I/O-bound tasks simultaneously.

# Multiprocessing | Process-based parallelism



# Concurrent.Futures | Launchinig parallel tasks

[concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html) module provides a high-level interface for asynchronously executing callables.

A *ThreadPoolExecutor* (*ProcessPoolExecutor*) manages a pool of threads (processes) as workers. The maximum number of parallel workers is bounded by the *max_workers* argument, default set to `min(32, os.cpu_count()+4)`.

```python
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=5)
```

With a *ThreadPoolExecutor* object named *executor*, you can provision it with multiple tasks (function calls), and then the executor will automatically schedule these tasks and assign them to workers. Each *submit()* call returns a [*Future*](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future) object, which allows you to control the execution status. 

```python
import time
def wait_a_while(seconds, index):
    '''
    say we want threads to wait for a while
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
```

Output:

```
[future 1] sleeping for 5.0 seconds ...
[future 2] sleeping for 2.0 seconds ...
[future 2] waken up after 2.0 seconds
[future 1] waken up after 5.0 seconds
[  main  ] future 1 returns 5.0
[  main  ] future 2 returns 2.0
all done!
```

We can see from the output that although future 2 woke up before future 1 did, in the main thread future 1 still blocked future 2 with its *result()* call until it returned. Also, *shutdown()* method with *wait* argument set to *True* made the main thread wait for two futures to finish. 

Now let's make things a little more complex by using an alternative method to provisioning tasks, named [*map()*](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor.map), and a context manager (*with* statement) to get a better control over unexpected exceptions during execution.

Note that a context manager looks like:

```python
with ThreadPoolExecutor(max_workers=5) as executor:
    # ... use the executor ...
```

is equivalent to

```python
executor = ThreadPoolExecutor(max_workers=5)
try:
    # ... use the executor ...
finally:
    executor.shutdown(wait=True)
```

Here is an example using *map()* method to handle a list of tasks.

```python
from concurrent.futures import ThreadPoolExecutor

def my_pow(base, power):
    '''
    say we want threads to run a stupid power function
    '''
    result = 1
    for i in range(power):
        result = result * base
    return result

base_list = [1, 2, 3, 4, 5, 6]
power_list = [2, 3, 4, 4, 3, 2]
with ThreadPoolExecutor(max_workers=5) as executor:
    # map(callable, arg1_list, arg2_list, ...)
    result_list = executor.map(my_pow, base_list, power_list)
print(list(result_list))
```

Output:

```
[1, 8, 81, 256, 125, 36]
```

[*concurrent.futures.as_completed()*](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.as_completed) is also useful to handle a task list, but its usage is not included in my post at present. Find more about it on the official doc. I'll only cover an example of using *as_completed()* function.

```python
import concurrent.futures
import urllib.request

URLS = ['http://www.foxnews.com/',
        'http://www.cnn.com/',
        'http://europe.wsj.com/',
        'http://www.bbc.co.uk/',
        'http://some-made-up-domain.com/']

# Retrieve a single page and report the URL and contents
def load_url(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as conn:
        return conn.read()

# We can use a with statement to ensure threads are cleaned up promptly
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    # Start the load operations and mark each future with its URL
    future_to_url = {executor.submit(load_url, url, 60): url for url in URLS}
    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            data = future.result()
        except Exception as exc:
            print('%r generated an exception: %s' % (url, exc))
        else:
            print('%r page is %d bytes' % (url, len(data)))
```

(**NOTICE** I think I haven't really understood this example. Maybe I'm writing something wrong here. Please read with caution.) You can also find this example [here](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor-example). This example offers us a fantastic way to mark the connections between futures and their corresponding argument. Since the concurrent.futures module runs threads asychronously and does not guarantee to return in order, without connections between futures and arguments, sometimes we have no idea which arguments the returned future results come from. So this is when what the example shows us comes in handy.

# Producer-Consumer Threading



# Logging | Logging facility for Python

[Basic Tutorial](https://docs.python.org/3/howto/logging.html#logging-basic-tutorial), [Advanced Tutorial](https://docs.python.org/3/howto/logging.html#advanced-logging-tutorial)

[logging](https://docs.python.org/3/library/logging.html) module supports a flexible event logging system. 

To get grips of the basic idea of logging, read the following two tables.

**Table 1**: best tools for desired tasks

| Task you want to perform                                     | The best tool for the task                                   |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| Display console output for ordinary usage of a command line script or program | [`print()`](https://docs.python.org/3/library/functions.html#print) |
| Report events that occur during normal operation of a program (e.g. for status monitoring or fault investigation) | [`logging.info()`](https://docs.python.org/3/library/logging.html#logging.info) (or [`logging.debug()`](https://docs.python.org/3/library/logging.html#logging.debug) for very detailed output for diagnostic purposes) |
| Issue a warning regarding a particular runtime event         | [`warnings.warn()`](https://docs.python.org/3/library/warnings.html#warnings.warn) in library code if the issue is avoidable and the client application should be modified to eliminate the warning[`logging.warning()`](https://docs.python.org/3/library/logging.html#logging.warning) if there is nothing the client application can do about the situation, but the event should still be noted |
| Report an error regarding a particular runtime event         | Raise an exception                                           |
| Report suppression of an error without raising an exception (e.g. error handler in a long-running server process) | [`logging.error()`](https://docs.python.org/3/library/logging.html#logging.error), [`logging.exception()`](https://docs.python.org/3/library/logging.html#logging.exception) or [`logging.critical()`](https://docs.python.org/3/library/logging.html#logging.critical) as appropriate for the specific error and application domain |

**Table 2**: explanation of event levels

| Level      | When it’s used                                               |
| :--------- | :----------------------------------------------------------- |
| `DEBUG`    | Detailed information, typically of interest only when diagnosing problems. |
| `INFO`     | Confirmation that things are working as expected.            |
| `WARNING`  | An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected. |
| `ERROR`    | Due to a more serious problem, the software has not been able to perform some function. |
| `CRITICAL` | A serious error, indicating that the program itself may be unable to continue running. |

Basic logging functions are named after the level or severity of the events they track. For example, *logging.info()* tracks `INFO`-level events, which show the program is working as expected.

Logging module tracks events based on their level or severity. The default level to track is `WARNING`, which means only events with level `WARNING` or a higher level will be recorded. For example,

```python
>>> import logging
>>> logging.warning("Watch out!")
WARNING:root:Watch out!
>>> logging.info("I told you so")
>>> 
```

*logging.info()* doesn't yield a command-line output in the example above because its level, `INFO`, is lower than `WARNING`, thus not recorded.

Below is an example of using basic logging functions in an interactive session. Remark that all functions used in the example can be put into a program to implement more complicated logics.

```python
>>> import logging
>>> logging.basicConfig(
      format="%(asctime)s [%(levelname)s] %(message)s",  # set logging format
      datefmt="%m/%d/%Y %I:%M:%S %p",  # set datetime format
      level=logging.DEBUG  # set track level, DEBUG means to display all event messages
    )
>>> logging.debug("This message should appear on the console")
03/05/2020 11:37:02 AM [DEBUG] This message should appear on the console
>>> logging.info("So should this")
03/05/2020 11:37:16 AM [INFO] So should this
>>> logging.info("And this, too")
03/05/2020 11:37:23 AM [INFO] And this, too
>>> 
```

Write event logging to a target file:

```python
>>> import logging
>>> logging.basicConfig(
      filename="example.log",  # write all logging to exampel.log
      filemode="w",  # open exampel.log in 'w' mode, write from the beginning
      level=logging.DEBUG  # set track level to DEBUG
    )
>>> logging.debug("now this file %s is open", "example.log")
>>> logging.warning("be %s not to write %s things", "cautious", "illegal") 
>>> logging.error("can\'t think of anything else to write")
>>> 

# $ cat example.log
# DEBUG:root:now this file example.log is open
# WARNING:root:be cautious not to write illegal things
# ERROR:root:can't think of anything else to write
```

# Itertools | Functions creating iterators for efficient looping

[itertools module](https://docs.python.org/3/library/itertools.html) implements a number of iterator building blocks.

First we need to clarify the concept of [iterable](https://docs.python.org/3/glossary.html#term-iterable). The official doc describes as an object that is capable of returning its members one at a time. Typical iterables:

- (all [sequence](https://docs.python.org/3/glossary.html#term-sequence) types): list, str, tuple, range, ...
- (some non-sequence types): dict, file objects, ...
- any user-defined classes with an [*\_\_iter\_\_()*](https://docs.python.org/3/reference/datamodel.html#object.__iter__) method or with a [*\_\_getitem()\_\_*](https://docs.python.org/3/reference/datamodel.html#object.__getitem__) method that implements [Sequence](https://docs.python.org/3/glossary.html#term-sequence) semantics. (better check out these concepts in the official doc, below is my personal simple summary)
  Generally, the *\_\_iter\_\_()* method returns a new [iterator](https://docs.python.org/3/glossary.html#term-iterator) that can iterate over all the objects in the container for sequence-like containers, or iterate over the keys for mapping-based containers (such as dict), and the *\_\_getitem()\_\_* method implements access to elements given integer keys or slice objects.

[**Iterables**](https://docs.python.org/3/glossary.html#term-iterable) are typically used in a [for](https://docs.python.org/3/reference/compound_stmts.html#for) loop and many other places where a sequence is needed, such as *zip()*, *map()*, and *enumerate()*. In most cases, these statements automatcially call the *iter()* function to obtain an iterator from these iterables.

For an [**iterator**](https://docs.python.org/3/glossary.html#term-iterator), the most important method is its [*\_\_next()\_\_* method](https://docs.python.org/3/library/stdtypes.html#iterator.__next__), which returns the next item in an iterable.

Alright, with adequate knowledge of iterables and iterators, let's dive into the functions provided by the itertools module. I believe you'll find them helpful and convenient.

[*islice()* method](https://docs.python.org/3/library/itertools.html#itertools.islice) makes an iterator that returns selected elements from a given *iterable*. It has two signatures:

- *itertools.islice(iterable, stop)*: returns an iterator over *iterable*[:stop]
- *itertools.islice(iterable, start, stop[, step])*: iterator over *iterable*[start:stop:step]

```python
>>> from itertools import islice
>>> iterable = "ABCDEFG"
>>> list(islice(iterable, 2))
['A', 'B']
>>> list(islice(iterable, 2, 4))
['C', 'D']
>>> list(islice(iterable, 2, None))
['C', 'D', 'E', 'F', 'G']
>>> list(islice(iterable, 2, None, 2))
['C', 'E', 'G']
```

[*starmap()* method](https://docs.python.org/3/library/itertools.html#itertools.starmap) makes an iterator that computes a given *function* using arguments obtained from a given *iterable*. Comparing with [*map()*](https://docs.python.org/3/library/functions.html#map) for which *function* arguments are organized as several iterables, one iterable for an argument, the *starmap()* method is used when *function* arguments are organized as a list of tuples. See the following example.

```python
>>> from itertools import starmap
>>> def my_mul(a, b):
...     return a*b
... 
>>> list(map(my_mul, [1, 2, 3], [2, 3, 4]))
[2, 6, 12]
>>> list(starmap(my_mul, [(1, 2), (2, 3), (3, 4)]))
[2, 6, 12]
```

