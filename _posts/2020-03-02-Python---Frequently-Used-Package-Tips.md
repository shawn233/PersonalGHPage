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



As a dedicated Python supporter, I use Python to handle miscellaneous programming tasks every day, from writing a simple scipt to modify file extension (e.g., .txt -> .md) in batches, to running neural network models.

A common dilemma for crazy Python programmers like me, is to remember how to use uncountable packages. For one task, I learned to use numpy, matplotlib, and scipy; for another task, I turned to the docs of pandas, pillow and pytorch.  But when I get back to matplotlib, I have to read the docs again because when I stop using a package, I tend to forget quickly. 

Now you see why I intend to write this post. In this post, I'll note down the core functions of frequently used packages, storing my knowledge about a package in the most compact way while I still can.

This post is helpful to me, bring up some of my faded memories. If you are interested, go ahead and read it! You'll surely find the part that works for you.

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
print(args.word) 		# harry
print(args.number) 		# 3
print(args.seperator)	# **
```

# Threading



# Multiprocessing



# Concurrent



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



 

# Itertools

