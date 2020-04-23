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

## Directory Traverse

Here are a few frequently used methods that support traversing directories.

- [*os.walk()*](https://docs.python.org/3/library/os.html#os.walk): display a directory tree, expanding all subdirectories, recursively
- [*os.scandir()*](https://docs.python.org/3/library/os.html#os.scandir): iterate over all entries (files and subdirectories) in a given directory
- [*os.listdir()*](https://docs.python.org/3/library/os.html#os.listdir): return a list containing names of all entries in a given directory

Let's say we have a folder that stores all of our school materials, named `lectures`, in which we have the following folders and files.

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

Now I'll show you three ways to traverse the `lectures` folder.

The first method is *os.listdir()*. It returns a list of all entry names in the first layer of the directory tree.

```python
>>> import os
>>> entries = os.listdir("./lectures/")
>>> entries
['Programming', 'syllabus.txt', 'Maths', 'English']
```

The second method is *os.scandir()*. It returns an iterator of [*os.DirEntry*](https://docs.python.org/3/library/os.html#os.DirEntry) objects corresponding to all entries in the first layer of the directory tree. So the results are similar to *os.listdir()* to some extent. *stat* info can also be accessed by  *DirEntry* objects.

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

The third method is *os.walk()*. It generates the whole directory tree, expanding all subdirectories. It is like a higher-level method which internally calls *os.scandir()*.

By setting the parameter *topdown* as `True`, this method runs a root-first traverse on the directory tree. The official docs say the *dirnames* can be changed in-place, e.g., *del* the *dirnames* list may prune the subsequent visits to subdirectories in this list.

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

# Shutil

[*copyfile()*]

[*rmtree()*]

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

# Zipfile | Work with ZIP archives

[zipfile module](https://docs.python.org/3/library/zipfile.html) allows Python programmers to deal with .zip archives. It supports the following operations:

- read an existing ZIP file: `mode='r'`.
- create and write a new ZIP file (truncate if file exists): `mode='w'`.
- append to and existing ZIP file: `mode='a'`.
- exclusively create and write a new file (raise a *FileExistError* if file exists): `mode='x'`.

```python
import zipfile
```

Say I have a bunch of text files organized in the structure below:

```bash
$ tree .
.
├── bar.txt
├── first
│   ├── second
│   │   └── yaa.txt
│   └── whoo.txt
└── foo.txt
```

Now I would like to compress `bar.txt` and `./first` folder into a ZIP file named `archive.zip`. This is how I will do it:

```python
>>> with zipfile.ZipFile("archive.zip", "w", compression=zipfile.ZIP_DEFLATED) as myzip:
...     myzip.write("bar.txt")
...     for root, dirs, files in os.walk("./first"):
...         for filename in files:
...             myzip.write(os.path.join(root, filename))
...
>>> with zipfile.ZipFile("archive.zip", "r") as myzip:
...     myzip.printdir()
... 
File Name                    Modified             Size
bar.txt                  2020-03-19 13:41:54         9
first/whoo.txt           2020-03-19 13:41:06         8
first/second/yaa.txt     2020-03-19 13:41:20         9
```

- Context manager (*with* statement) here is equivalent to

  ```python
  myzip = zipfile.ZipFile("archive.zip", "w")
  # ... my.write(file) ...
  myzip.close()
  ```

- The *compression* argument indicates the ZIP compression method to use. By default, its value is [*zipfile.ZIP_STORED*](https://docs.python.org/3/library/zipfile.html#zipfile.ZIP_STORED), which doesn't compress at all and results in an uncompressed archive. By setting *compression* as [*zipfile.ZIP_DEFLATED*](https://docs.python.org/3/library/zipfile.html#zipfile.ZIP_DEFLATED), we choose the usual ZIP compression method. If we use the *ZIP_DEFLATED* method, then we can further set the *compressionlevel* argument to control the compression level.

- This example is just for demonstration. In practice, it is unwise to compress a lot of files of small sizes, because it will almost likely result in a ZIP file whose size is much larger than the total file sizes. (In this example, I actually got a ZIP file of size 362 bytes.)

We have already created a ZIP file named `archive.zip`. Say if we would like to add another text file to the archive, called `foo.txt`, we can use the append (`'a'`) mode.

```python
>>> with zipfile.ZipFile("archive.zip", "a", compression=zipfile.ZIP_DEFLATED) as myzip:
...     myzip.write("foo.txt")
...     myzip.printdir()
... 
File Name                    Modified             Size
bar.txt                  2020-03-19 13:41:54         9
first/whoo.txt           2020-03-19 13:41:06         8
first/second/yaa.txt     2020-03-19 13:41:20         9
foo.txt                  2020-03-19 13:41:40         9
>>>
```

Now, the ZIP file `archive.zip` contains the complete content and structure of our files. We can decompress it and restore all these files, by simply changing to the read (`'r'`) mode and invoking the [*extract()*](https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile.extract) / [*extractall()*](https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile.extractall) methods.

# Numpy

```python
import numpy as np
```

## Combinig arrays

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

## Sampling

[*choice()*](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.choice.html#numpy-random-choice) method in the numpy.random module generates a random sample from a given 1-D array.

```python
>>> # sample from a range
>>> np.random.choice(8, size=8, replace=True)
array([1, 0, 1, 7, 4, 2, 3, 5])
>>> np.random.choice(8, size=8, replace=False)
array([7, 5, 3, 0, 4, 1, 2, 6])
>>> np.random.choice(8, size=(2, 3), replace=True)
array([[0, 4, 2],
       [6, 4, 7]])
>>>
>>> # sample from an array
>>> friends = ['cat', 'dog', 'rabbit', 'bunny']
>>> np.random.choice(friends, size=3, replace=False)
array(['dog', 'cat', 'bunny'], dtype='<U6')
>>> np.random.choice(friends, size=3, replace=True)
array(['bunny', 'bunny', 'cat'], dtype='<U6')
```

- If the first parameter is an *int* such as 8, samples are generated from the range `np.arange(8)`, which is a list, `[0, 1, 2..., 7]`.
- parameter *size* indicates the output shape.
- parameter *replace* indicates whether the sample is with or without replacement.

# Pandas



# Matplotlib



# PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.Functional as F
from torch.utils.data import Dataset, Dataloader
from torchvision import transforms
```

## Loss Computation

[*torch.nn.Softmax*](https://pytorch.org/docs/stable/nn.html#torch.nn.Softmax) class applies the softmax function to rescale Tensors

- output shape = input shape
- In the output Tensor, along a given dimension *dim*, all element values lie in the range [0, 1] and sum to 1

$$
\text{Softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^n\exp(x_j)}
$$

```python
>>> layer = nn.Softmax(dim=1) # along dim 1 elements sum to 1
>>> input = torch.randn(2, 3)
>>> input
tensor([[ 1.8528,  1.2640, -0.0431],
        [-0.9120, -1.7943, -0.6436]])
>>> output = layer(input)
>>> output
tensor([[0.5864, 0.3255, 0.0881],
        [0.3674, 0.1520, 0.4805]])
```

[*torch.nn.NLLLoss*](https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss) is the **negative log likelihood** loss.

[*torch.nn.CrossEntropyLoss*](https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss) combines *nn.LogSoftmax* and *nn.NLLLoss* in one single class. Now we discuss how to use these two loss criterions to compute loss.

Say we have an image classification model that tells whether an image is a cat, dog, or pig. Now, we train this net with a set of images, within which we randomly pick three, and get the following extracted features (each line represents an image):

```python
>>> pred
tensor([[-1.0395,  1.1958, -0.3859],
        [-1.9872, -0.2764, -1.6573],
        [ 2.1350, -0.2390,  0.7698]])
```

We know that the labels for these images are [cat, dog, pig], represented as [0, 1, 2] with numbers.

```python
target = torch.tensor([0, 1, 2])
```

Since *pred* is the direct output of the last layer, its values don't represent probabilities, but will do after getting processed by the softmax layer.

Now let's directly compute the loss.

```python
>>> # 1. process features to probabilities by softmax
>>> # set dim to 1 because we want each line sums up to 1
>>> softmax = nn.Softmax(dim=1)
>>> # the net considers images as [dog, dog, cat]
>>> softmax(pred)
tensor([[0.0815, 0.7619, 0.1567],
        [0.1262, 0.6983, 0.1755],
        [0.7416, 0.0691, 0.1894]])
>>> # 2. get log likelihood
>>> pred_ll = torch.log(softmax(pred))
>>> pred_ll
tensor([[-2.5073, -0.2720, -1.8537],
        [-2.0699, -0.3591, -1.7400],
        [-0.2990, -2.6729, -1.6641]])
>>> # 3. obtain log likelihoods of each ground truth label
>>> truth_ll = pred_ll[torch.arange(3), target]
>>> truth_ll
tensor([-2.5073, -0.3591, -1.6641])
>>> # 4. the loss is
>>> torch.mean(-truth_ll)
tensor(1.5102)
```

Remark that loss becomes zero when and only when probabilities, or softmax(pred), become

```python
>>> truth
tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]])
```

Now we see computing loss is actually quite simple. If we summary the codes above with one line, we get

```python
loss = torch.mean(
    -torch.log(
        F.softmax(
            pred,
            dim=1
        )
    )[
        torch.arange(pred.size()[0]),
        target
    ]
)
```

Actually because I use *torch.nn.functional.softmax* here, this piece of code only works for 2 dimension cases. With a wider application, pyTorch makes this process quite simple and flexible by pre-defining many types of loss creiterions. For example, we can get the same loss with the following methods.

```python
>>> # 1. apply NLL loss to log likehood 
>>> nllloss = nn.NLLLoss()
>>> nllloss(pred_ll, target)
tensor(1.5102)
>>> # 2. apply log Softmax and then NLL loss to net output 
>>> logsoftmax = nn.LogSoftmax(dim=1)
>>> nllloss(logsoftmax(pred), target)
tensor(1.5102)
>>> # 3. directly apply Cross Entropy loss to net output
>>> crossentropy = nn.CrossEntropyLoss()
>>> crossentropy(pred, target)
tensor(1.5102)
```

Also remark all the criterions mentioned above have corresponding functions in the *torch.nn.functional* module. Specifically:

- *nn.Softmax*: *F.softmax()*
- *nn.LogSoftmax*: *F.log_softmax()*
- *nn.NLLLoss*: *F.nll_loss()*
- *nn.CrossEntropyLoss*: *F.cross_entropy()*

## Model Restoration

[official doc](https://pytorch.org/docs/stable/notes/serialization.html): simple but not enough

```python
torch.save(the_model.state_dict(), PATH)
# later
the_model = TheModelClass(*args, **kwargs)
the_model.load_state_dict(torch.load(PATH))
```

Saving and restoring models mainly rely on two functions defined in the torch module: (1) [*torch.save()*](https://pytorch.org/docs/stable/torch.html#torch.save); (2) [*torch.load()*](https://pytorch.org/docs/stable/torch.html#torch.load).

[*torch.save()*](https://pytorch.org/docs/stable/torch.html#torch.save) saves an object (any Python object) to a disk file. It's remarkable that any object can be saved with this method.

```python
>>> fpath = "tmp.pt"
>>> a = [0, 1.34, "hello world", ("a", "b")]
>>> torch.save(a, fpath)
```

[*torch.load()*](https://pytorch.org/docs/stable/torch.html#torch.load) loads an object saved with *torch.save()* from a file.

```python
>>> torch.load(fpath)
[0, 1.34, 'hello world', ('a', 'b')]
```

Therefore, torch module actually provides a universal set of methods to save and later restore any objects. Well, at most times, they are used for saving model parameters, but with these methods we are allowed to save in our own ways.

Here I provide a clear way to save parameters along with other model information in one file with a *dict*. [reference](https://zhuanlan.zhihu.com/p/38056115)

```python
torch.save(
    {
        'epoch': epoch+1,
        'state_dict': model.state_dict(),
        'best_loss': min_loss,
        'optimizer': optimizer.state_dict(),
        'alpha': loss.alpha, # no idea what it means
        'gamma': loss.gamma
    },
    meaningful_path_with_train_info
)

model_ckpt = torch.load(the_path)
model.load_state_dict(model_ckpt['state_dict'])
optimizer.load_state_dict(model_ckpt['optimizer'])
```

## Dataset loading utility

```python
torch.utils.data.DataLoader(
    dataset, 
    batch_size=1, 
    shuffle=False,
    sampler=None, 
    batch_sampler=None, 
    num_workers=0, 
    collate_fn=None,
    pin_memory=False, 
    drop_last=False, 
    timeout=0,
    worker_init_fn=None
)
```

- *dataset*: a map/iterable-style dataset
- *batch_size*: size of each batch
- *shuffle*: set as *False* to construct a sequential sampler, or *True* to construct a shuffled sampler
- *sampler*: a user-specified custom sampler that is different from the sequential or shuffled sampler.
- *batch_sampler*: a custom sampler that yields a list of batch indices at a time.
- *num_workers*: default to 0 (single-process data loading). Setting it as a positive value will turn on multi-process data loading.
- *collate_fn*: a callable object to collate lists of samples into batches (collate: to assemble in proper order).
- *pin_memory*: set as *True* to put the fetched data Tensors in pinned (page-locked) memory, and thus enable faster data transfer to CUDA-enabled GPUs.
- *drop_last*: True to drop out the last incomplete batch.

In the most common case, when we use [**automatic batching**](https://pytorch.org/docs/stable/data.html#automatic-batching-default), the data loader roughly works like:

```python
for indices in batch_sampler:
    yield collate_fn([dataset[i] for i in indices])
```

- Since the loader uses a batch sampler, so for each time the sampler returns a list of indices.

When your model is designed to process single samples, or when each data sample is required to be processed independently, you can [disable automatic batching](https://pytorch.org/docs/stable/data.html#disable-automatic-batching) by setting both *batch_size* and *batch_sampler* to *None*. Then the data loader works like:

```python
for index in sampler:
    yield collate_fn(dataset[index])
```

Concepts

- Dataset types
  - [Map-style datasets](https://pytorch.org/docs/stable/data.html#map-style-datasets): subclass of [*torch.utils.data.Dataset*](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) that implements *\_\_getitem\_\_()* and *\_\_len\_\_()* protocols. Such a dataset supports sample accessing by indexing.
  - [Iterable-style datasets](https://pytorch.org/docs/stable/data.html#iterable-style-datasets): subclass of [*torch.utils.data.IterableDataset*](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) that implements the *\_\_iter\_\_()* protocol. Such a dataset, when called `iter(dataset)`, returns an iterable stream of data reading from a database, a remote server, or even logs generated in real time.
- [Sampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler): an iterable object over the **indices** to datasets, not the data samples themselves.

Examples

```python

```

## Image transformation



# Pillow



# Argparse | Parse for command-line options, arguments and sub-commands

```python
import argparse
```

argparse is a frequently used package that supports convenient management of arguments passed from the command line. It supports several types of arguments, including:

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

## Simple options

To start with defining argument options, create an [*ArgumentParser*](https://docs.python.org/3.7/library/argparse.html#argumentparser-objects) object. You may add a proper description to help users understand your program.

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

Now, let's come to two other use cases of argparse module.

## Boolean option

A boolean option takes no value in the command line. For example, in most commands we have a `-v / --verbose` option. If we use this option, command execution will output more information than usual.

How to implement this type of option? In the *add_argument()* method, we can pass a parameter named [*action*](https://docs.python.org/zh-cn/3/library/argparse.html#action):

```python
>>> parser.add_arguemnt("-v", "--verbose", help="more info",
                   action="store_true")
```

Since this argument is set with the `store_true` *action*, it takes the following default values:

- *dest*: verbose
- *default*: False

In another word, if this option is not used, `args.verbose` defaults to *False*.

```python
>>> args = parser.parse_args([])
>>> args.verbose
False
```

Instead, if it is, we have

```python
>>> args = parser.parse_args("-v".split())
>>> args.verbose
True
```

Now we can implement a boolean option with the help of *action* parameter. Actually, this parameter can take more values, as listed below.

- *action*=`store`: default *action*. The argument holds a value taken from command line.
- *action*=`store_false`: The argument takes no value from the command line. If the option is used, argument is set to False, otherwise True.
- *action*=`store_const`: The argument works with another parameter named *const*. While taking no value from the command line, the argument is set to the value of *const* if the option is used. 

## Argument list

Sometimes we would like to pass a list of values to an argument option. There are more than one ways to implement this, but here I only provide one that I think as the most convenient.

```python
>>> parser.add_argument("-l", "--list", help="integer list",
                       nargs="+", type=int, action="store")
```

- If the option is not used, `args.list` is set to *None*, the default value of parameter *default* in *add_argument()* method.

```python
>>> args = parser.parse_args("--list 1 2 3 4".split())
>>> args.list
[1, 2, 3, 4]
```

The trick here is that we use the parameter [*nargs*](https://docs.python.org/zh-cn/3/library/argparse.html#nargs) in the *add_argument()* method. *nargs* takes the following values:

* an integer *N*: this option consumes exactly *N* values in the command line. Note that if *N=1*, the option still produces a single-element list.
* `?`: this option takes one value from the command line if possible, and produces a single object. If no value is available, use the *default* value.
* `*` / `+`: this option takes as many values as possible. The difference between `*` and `+` is that `+` must take at least one value, otherwise it raises an exception.
* `argparse.REMAINDER`: take all remaining values.

# Threading | Thread-based parallelism

The [threading module](https://docs.python.org/3/library/threading.html) allows you to implement **parallelism** in a convenient way.

```python
import threading
```

(Using the threading module usually implies a lot of codes, including worker definition, thread creation, and main thread implementation. So due to restricted space, sample codes in this section are mostly provided as pseudocodes. But don't worry, these codes are fully commented and clear enough to get the idea. For more curious readers, I also uploaded some [realistic samples](https://github.com/shawn233/shawn233.github.io/tree/master/_posts/related-codes/threading-samples) that you can play with.)

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

[LogRecord attributes](https://docs.python.org/3/library/logging.html#logrecord-attributes)

![image-20200413230054283](https://github.com/shawn233/shawn233.github.io/raw/master/_posts/.assets/image-20200413230054283.png)

![image-20200413230112508](https://github.com/shawn233/shawn233.github.io/raw/master/_posts/.assets/image-20200413230112508.png)

# Itertools | Functions creating iterators for efficient looping

[itertools module](https://docs.python.org/3/library/itertools.html) implements a number of iterator building blocks.

First we need to clarify the concept of [iterable](https://docs.python.org/3/glossary.html#term-iterable). The official doc describes it as an object that is capable of returning its members one at a time. Typical iterables:

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

# Collections

# Time | Time access and conversions

```python
import time
```

Term explanation

- **epoch**: the point where the time starts, can be accessed via `time.gmtime(0)`.
- **UTC**: Coordinated Universal Time (can be interchanged with GMT, Greenwhich Mean Time).
- **DST**: Daylight Saving Time.
- The precision of the various real-time functions may be less than suggested by the units in which their value or argument is expressed. E.g. on most Unix systems, the clock “ticks” only 50 or 100 times a second.
- ![image-20200413191921394](https://github.com/shawn233/shawn233.github.io/raw/master/_posts/.assets/image-20200413191921394.png)

## Formatted Time

By default, a time is represented as a timestamp, or float. To convert such a timestamp to a formatted time string such as 

```bash
2020-04-13 (Apr,Mon) 09:47:13 PM CST
```

, class [*time.struct_time*](https://docs.python.org/3/library/time.html#time.struct_time) and method [*time.strftime()*](https://docs.python.org/3/library/time.html#time.strftime) come in handy.

A [*struct_time*](https://docs.python.org/3/library/time.html#time.struct_time) object represents a specific moment in time, with values stored as a named tuple.

![image-20200413215115891](https://github.com/shawn233/shawn233.github.io/raw/master/_posts/.assets/image-20200413215115891.png)

With a *struct_time* object, you can display it, i.e. convert it to a time string, using [*strftime()*](https://docs.python.org/3/library/time.html#time.strftime). The output format can be passed as a parameter, which supports directives as listed below.

![image-20200413215826101](https://github.com/shawn233/shawn233.github.io/raw/master/_posts/.assets/image-20200413215826101.png) 

![image-20200413215848446](https://github.com/shawn233/shawn233.github.io/raw/master/_posts/.assets/image-20200413215848446.png)

![image-20200413215917460](https://github.com/shawn233/shawn233.github.io/raw/master/_posts/.assets/image-20200413215917460.png)

A *struct_time* object is usually obtained from one of the following methods, as the return value:

- [*gmtime()*](https://docs.python.org/3/library/time.html#time.gmtime): converts seconds since the epoch to a *struct_time* object in UTC. 
- [*localtime()*](https://docs.python.org/3/library/time.html#time.localtime): converts seconds since the epoch to a *struct_time* object in the local timezone.
- [*strptime()*](https://docs.python.org/3/library/time.html#time.strptime): parses a time string according to a given format.

For example, the formatted string at the section beginning is actually obtained from the method *localtime()*.

```python
>>> time.strftime("%Y-%m-%d (%b,%a) %I:%M:%S %p %Z", time.localtime())
'2020-04-13 (Apr,Mon) 09:47:13 PM CST' 
```

Now this is all for a *time* implementation. Similar functionality is also supported in the *datetime* module. You may check it out in the [official docs](https://docs.python.org/3/library/datetime.html).

# Datetime

```python
from datetime import *
```

The [datetime](https://docs.python.org/3/library/datetime.html) module supplies several classes for manipulating dates and times.

Subclass relationships:

```python
object
    timedelta # difference between date/time/datetime 
              # instances (unit: microsecond)
    tzinfo # time-zone information
        timezone
    time # independent of any particular day
    date # a naive date
        datetime
```

Similar(For curious readers) Distinguishing **aware** and **naive** objects ([official docs](https://docs.python.org/3/library/datetime.html#aware-and-naive-objects)):

- An **aware** object is aware of its timezone. It represents a specific moment in time that is not open to interpretation (no ambiguity). Specifically, a time or datetime object `obj` ("Date objects are always naive.") is aware if both of the following hold, otherwise naive.
  - `obj.tzinfo` is not *None*.
  - `obj.tzinfo.utcoffset(d)` does not return *None*.
- A **naive** object, in contrast, is not subject to any particular timezone. It can't be unambiguously interpreted to a specific moment in time unless further explained by an application. For example, a naive object may be interpreted as local time, UTC (Coordinated Universal Time), or time in some other timezone. "Naive objects are easy to understand and to work with, at the cost of ignoring some aspects of reality."

# Psutil

# Chardet

# Re | Regular expression operations

The [re](https://docs.python.org/3/library/re.html) module provides regular expression matching operations.

```python
import re
```

## Basics

In this section you will learn what is regular expressions and how to write a regular expression.

- A **regular expression** specifies a set of strings that matches it.

- A **character class** is a set of characters that you wish to match, enclosed by `[` and `]`.
  - `[abc]` or `[a-c]` matches any of `a`, `b`, or `c`.
  - `[^5]` matches any character except `'5'`.
  - **No metacharacter is active** inside a character class. For `^`, it only indicates a complementing set when it appears as the first character, otherwise it means a match to a literal `^`.
  - Special sequences: predefiened sets of characters
    - decimal digits: `\d = [0-9]`
    - whitespace characters: `\s = [\t\n\r\f\v]`
    - alphanumeric characters: `\w = [a-zA-Z0-9_]`
    - ...
    - Special sequences can be included inside a character class.
  
- **Dot `.` matches anything except a newline character (`\n`)**. But in the `re.DOTALL` mode, `.` matches anything including `\n`.

- **Greedy Repeated qualifiers**: `* + ? {m,n}`, "greedy" means they'll consume as much text as possible. 
  
  - `{m,n}`: repeat at least `m`, at most `n` times. Omitting `m` is interpreted as a lower limit of 0, while omitting `n` results in an upper bound of infinity.
  - `* = {0, }`: zero or more times. Note using `*` matches the empty string. For example, `'a*'` matches `''`, `'a'`, `'aa'`, `'aaa'` and so forth.
  - `+ = {1, }`: one or more times.
  - `? = {0,1}`: once or zero times.
  - The corresponding non-greedy set: `*? +? ?? {m,n}?`. For example, to match an HTML tag, you'd better use `<.*?>` instead of `<.*>`. (It's not recommended to use regular experssions to parse HTML or XML in the [official doc](https://docs.python.org/3/howto/regex.html#greedy-versus-non-greedy).)
  
- **Groups** in a regular expression, indicated with `(` and `)`, capture substrings of interest. Matching results can be retrieved by passing a group index to *group()*, *start()*, *end()*, and *span()*.

  - Group 0 is always the whole RE. Other group indices are determined by counting the opening parenthesis from left to right. For example, `0 (1 (2) ) (3)`.

  - An example of retrieving group results

    ```python
    >>> p = re.compile('(a(b)c)d')
    >>> m = p.match('abcd')
    >>> m.group(0)
    'abcd'
    >>> m.groups() # start from group 1
    ('abc', 'b')
    >>> m.group(1)
    'abc'
    >>> m.group(2)
    'b'
    ```

  - **Backreference**: `\1` will succeed if the exact contents of group 1 can be found at the current position. This syntax is frequently used in string subsitutions.

- **Group extensions**: `(?...)` is introduced by Perl developers as an extension syntax. The specific extension is indicated by the characters immediately after `?`: 

  - `(?:...)`: Non-capturing group, which doesn't have an index and thus doesn't allow retrieval from the result.
  - `(?=...)`: Positive lookahead assertion, see [HOWTO](https://docs.python.org/3/howto/regex.html#lookahead-assertions).
  - `(?!...)`: Negative lookahead assertion, see [HOWTO](https://docs.python.org/3/howto/regex.html#lookahead-assertions).
  - `(?P...)`: Python-specific extensions, as follows.
  - `(?P<name>...)`: Named group, which additionally associates a name reference with the group. (Method [*gorupdict()*](https://docs.python.org/3/library/re.html#re.Match.groupdict) retrieves all named groups as a *dict*.)
  - `(?P=name)`: Backreference for the named group, which means the contents of the group called *name* should again be matched at the current position.

Examples

- domain name (`**.**.**`): `^[\w-]+(?:\.[\w-]+)+$`

- email addresses (`**@**.**`): `^[\.\w-]+@[\w-]+(?:\.[\w-]+)+$`

- (utf-8) Chinese characters: `[\u4e00-\u9fa5]`

- file name extension, excluding .bat and .exe (very hard, isn't it):

  ```
  .*[.](?!bat$|exe$)[^.]*$
  ```

## Python RE

An example of matching correctly-spelt names:

Say we want to match all names of celebrities out of the next-year Grammy Award's introduction.

> Ladies and gentlemen! Let me introduce you: Shawn Mendes, Ed Sheeran and T. Swift!

```python
>>> import re
>>>
>>> # 1. Start from your desired matches
>>> names = ["Lebron James", "K. Bryant", "Harley Quinn"]
>>> #    find the shared pattern
>>> name_spelt_correctly = r"[A-Z][a-z\.]* [A-Z][a-z\.]*"
>>>
>>> # 2. Compile your pattern
>>> pattern = re.compile(name_spelt_correctly)
>>> #    see if it works perfectly
>>> pattern.match("Lebron James, the big guy")
<re.Match object; span=(0, 12), match='Lebron James'>
>>> pattern.search("dear Saoise Ronan")
<re.Match object; span=(5, 17), match='Saoise Ronan'>
>>> pattern.match("AtOTALLY wrongname")
>>> # returns None if pattern can't match
>>>
>>> # 3. Do the match!
>>> intro = "Ladies and gentlemen! Let me introduce you: Shawn Mendes, Ed Sheeran and T. Swift!"
>>> pattern.findall(intro)
['Shawn Mendes', 'Ed Sheeran', 'T. Swift']
>>> #    and it works
```

The above example is not perfect, since it goes wrong when it sees "Hello Rachel Mcadams!", or "Ms. Jennifer Lawrence". Anyway, this example reveals the pipeline of using regular expressions in Python.

- Write a pattern string up to your demand.

- Compile the pattern string and get a [*Pattern* object](https://docs.python.org/3/library/re.html#regular-expression-objects), by [*re.compile()*](https://docs.python.org/3/library/re.html#re.compile).

- Perform a matching operation and get your result.

  - [*search()*](https://docs.python.org/3/library/re.html#re.search): scan through *string*, and return the first match, or *None* if no match is found.

    ```python
    >>> re.search(r"\d+", "a12b34c")
    <re.Match object; span=(1, 3), match='12'>
    >>> re.search(r"\d+", "abc")
    >>> # None
    ```

  - [*match()*](https://docs.python.org/3/library/re.html#re.match): return a match from the beginning of *string*, or *None* if the beginning of the *string* doesn't match the pattern.

    ```python
    >>> re.match(r"\d+", "123---")
    <re.Match object; span=(0, 3), match='123'>
    >>> re.match(r"\d+", "-123")
    >>> # None
    ```

  - [*findall()*](https://docs.python.org/3/library/re.html#re.findall): return a list of all non-overlapping matches of the pattern in *string*.

    ```python
    >>> re.findall(r"\d+", "1qaz23")
    ['1', '23']
    >>> re.findall(r"\d+", "qaswed")
    []
    ```
  
- Alternatively, a compiled pattern can be used to modify *string*:

  - [*split()*](https://docs.python.org/3/library/re.html#re.split): split *string* by the occurrences of the pattern. Any capturing groups will also appear in the result.
  
    ```python
    >>> re.split(r'[\W]+', 'Words, words, words.')
    ['Words', 'words', 'words', '']
    >>> re.split(r'([\W]+)', 'Words, words, words.')
    ['Words', ', ', 'words', ', ', 'words', '.', '']
    ```
  
  - [*sub(repl, string)*](https://docs.python.org/3/library/re.html#re.sub): return a new string obtained by replacing left-most occurences of the pattern in *string* by the replacement *repl*. Backreference can be used in *repl*, which allows you to incorporate portions of the original text in the resulting replacement string.
  
    ```python
    >>> p = re.compile('(blue|white|red)')
    >>> p.sub('colour', 'blue socks and red shoes')
    'colour socks and colour shoes'
    >>>
    >>> # An example using backreference
    >>> p = re.compile('section{ ( [^}]* ) }', re.VERBOSE)
    >>> p.sub(r'subsection{\1}','section{First} section{second}')
    'subsection{First} subsection{second}'
    >>>
    >>> # Another example using function as `repl`
    >>> def hexrepl(match):
    ...     "Return the hex string for a decimal number"
    ...     value = int(match.group())
    ...     return hex(value)
    ...
    >>> p = re.compile(r'\d+')
    >>> p.sub(hexrepl, 'Call 65490 for printing, 49152 for user code.')
    'Call 0xffd2 for printing, 0xc000 for user code.'
    ```

## Compilation Flags

![image-20200423094110464](https://github.com/shawn233/shawn233.github.io/raw/master/_posts/.assets/image-20200423094110464.png)

* multiple flags are concatenated with OR (`|`), such as `re.A | re.I`.
* `re.VERBOSE` makes your RE much more readable by taking the following effects:
  * Any whitespace outside a character class (`[ ]`) is ignored.
  * Supporting comments with a `#`.

Check out the [official HOWTO](https://docs.python.org/3/howto/regex.html#compilation-flags) for a detailed explanation.

## Appendix

Special Note 1: When a pattern is a unicode string (type *str*), the search string can be either a unicode (*str*) or a byte string (type *bytes*). But strings in different types are not a match. 

Special Note 2: DO use **raw** string notations for a regular expression pattern, with a prefix `'r'`. A normal pattern string has torepresent a backslash `\` as double backslashes `\\` subject to the rule of escape sequences, in which  `\\\\\` represents matching a literal backslash. But in raw strings, backslash `\` is not considered as part of an escape sequence, but as itself. So raw pattern strings avoid all these troubles.

Metacharacters:

```
. ^ $ * + ? { } [ ] \ | ( )
```

- `|`: `A|B` will match any string that matches either A or B.

  - Very low precedence: `Crow|Servo` matches either `'Crow'` or `'Servo'`.

- `^ $ \A \Z`: 
  If *re.MULTILINE* is not set,

  - `^` or `\A`: the beginning of the string.
  - `$` or `\Z`: the end of the string.

  If *re.MULTILINE* is set,

  - `^` also matches the beginning of each line (immediately after `\n`).
  - `$` also matches the end of each line (any location followed by `\n`).

- `\b`: word boundar (whitespace or non-alphanumeric character). For example, `\bclass\b` matches `'class'` in "no class at all" but not in "the declassified algorithm".  

  - MUST be used in a raw string, to avoid collision with Python's backspace (`\b`, ASCII value 8)

- `\D = [^\d] \S = [^\s] \W = [^\w] \B = [^\b]`.

# Requests

The [requests](https://requests.readthedocs.io/en/master/) library is the de facto standard for making HTTP requests in Python. It's elegant and simple. And as said in its official docs,

> **Requests** is an elegant and simple HTTP library for Python, built for human beings. 

## Sending requests

## Inspecting response

## Authentication

## Configuration



Reference: [Real Python](https://realpython.com/python-requests/)

## HTTP tips

Request header

```
[Method] [URL] [Protocol]\r\n
[FIELD-NAME 1]: [FIELD 1]\r\n
...
[FIELD-NAME N]: [FIELD N]\r\n
\r\n
[body]
```

| Method  | Meaning |
| ------- | ------- |
| GET     | requests a representation of the specified resource |
| POST    | sends data to the server |
| HEAD    | requests the header of a GET request |
| PUT     | creates a new resource or replaces a current resource with the request payload |
| DELETE  | deletes the specified resource |
| CONNECT | starts two-way communications with the requested resource |
| OPTIONS | describes the communication options for the target resource |
| TRACE   | performs a message loop-back test along the path to the target resource, for debug |
| PATCH   | applies partial modifications to a resource |

- GET request has no body
- As described in HTTP/1.1 specification, `POST` is designed to allow a uniform method to cover the following functions:
  ![image-20200418122036436](https://github.com/shawn233/shawn233.github.io/raw/master/_posts/.assets/image-20200418122036436.png)
- The response of a HEAD request has no body, but may have Content-Length, which relates to the GET request.
- A Put request is responded with 201 (Created) if the target resource is created, and with 202 (OK) or 204 (No Content) if the target resource is updated.
- An OPTIONS request can be used to identify allowed request methods. Its response has a `Allow` field with the allowed methods.
- An HTTP method is [**idempotent**](https://developer.mozilla.org/en-US/docs/Glossary/Idempotent) if an identical request can be made once or several times in a row with the same effect while **leaving the server in the same state**. In other words, an idempotent method should not have any side-effects (except for keeping statistics). Implemented correctly, `GET`, `HEAD`, `PUT` and `DELETE` method are idempotent, but not the `POST` method. To be idempotent, only the actual back-end state of the server is considered, the status code returned by each request may differ, such as `DELETE`.
- An HTTP method is [**safe**](https://developer.mozilla.org/en-US/docs/Glossary/safe) if it doesn't alter the state of the server. In other words, a method is safe if it leads to a read-only operation. Safe HTTP methods: `GET`, `HEAD` and `OPTIONS`. All safe methods are idempotent, but not all idempotent methods are safe, such as `PUT` and `DELETE`.

Reference
- [Status codes](https://en.wikipedia.org/wiki/List_of_HTTP_status_codes)
- [Header fields](https://developer.mozilla.org/zh-CN/docs/Web/HTTP/Headers)
- [MIME types](https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types/Common_types)

Response header

```
[Protocol] [Status]\r\n
[FIELD-NAME 1]: [FIELD 1]\r\n
...
[FIELD-NAME N]: [FIELD N]\r\n
\r\n
<html>\r\n
  ...
</html>
```

HTTP/1.0 drawback: each TCP connection only allows one request. N resources require N connections, which cause high connection overhead.

# Curses | Terminal handling for character-cell displays

```python
import curses
```

The [curses](https://docs.python.org/3/library/curses.html) module, as the de-facto standard for portable advanced terminal handling, provides a highly flexible terminal display API, based on abstract concepts such as window, pad, and textpad.

A curses-based application needs initilization, setting terminal to the right mode via special control codes, and destruction, restoring the terminal. The module provides us with a concise function to do all these things: [*curses.wrapper()*](https://docs.python.org/3/library/curses.html#curses.wrapper).

```python
def main(stdscr):
    # stdscr: standard screen
    # ... do something with stdscr ...
    
curses.wrapper(main)
```

The *wrapper()* function is roughly equivalent to

```python
def wrapper(func):
    # func is an callable object
    
    # initializes the standard screen object,
    # use control codes to set the terminal
    stdscr = curses.initscr() 
    
    # turns off automatic echoing of keys to the screen
    curses.noecho()
    
    # turns on the cbreak mode: react to key strokes instantly
    curses.cbreak()
    
    # receives special keys as escape sequences and
    # interprets them as special values
    # e.g. <left> -> curses.KEY_LEFT
    stdscr.keypad(True)
    
    try:
        return func(stdscr) # raise if any exception occurs
    finally:
        # always restore the terminal settings 
        curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        curses.endwin() #reverse curses.initscr() settings
```

Inside a wrapper, or in your main procedure, you can use the various functionalities supported by the curses module, to customize your curses application.

The most important concept in curses is the *Window*.

>  A **window** object represents a rectangular area of the screen, and supports methods to display text, erase it, allow the user to input strings, and so forth.

To get a window object, simply use the default window returned by [*curses.initscr()*](https://docs.python.org/3/library/curses.html#curses.initscr), or call [*curses.newwin()*](https://docs.python.org/3/library/curses.html#curses.newwin). (reference: [Window object docs](https://docs.python.org/3/library/curses.html#window-objects)) Inside a window, you can call [*window.subwin()*](https://docs.python.org/3/library/curses.html#curses.window.subwin) to get a sub-window. The behavior of a sub-window is totally the same as a window.

After you've got a window object. You can (usage: *window.method()*)

- Output (paint): 
  - [*addch()*](https://docs.python.org/3/library/curses.html#curses.window.addch): paint a character, and **overwrite** any character that already exists, at a given position.
  - [*addstr()*](https://docs.python.org/3/library/curses.html#curses.window.addstr) / [*addnstr()*](https://docs.python.org/3/library/curses.html#curses.window.addnstr): paint and overwrite a character string.
  - [*insch()*](https://docs.python.org/3/library/curses.html#curses.window.insch): insert a character, and **shift right** any characters that already exist, at a given position.
  - [*insertln()*](https://docs.python.org/3/library/curses.html#curses.window.insertln) / [*insdelln()*](https://docs.python.org/3/library/curses.html#curses.window.insdelln): insert blank lines and shift down lines that already exist.
  - [*insstr()*](https://docs.python.org/3/library/curses.html#curses.window.insstr) / [*insnstr()*](https://docs.python.org/3/library/curses.html#curses.window.insnstr): insert and shift right a character string.
- Input (get):
  - [*getch()*](https://docs.python.org/3/library/curses.html#curses.window.getch): get a character and return an integer (may not fall into the ASCII range because of special values). In no-delay mode, return -1 if no input, otherwise block to read.
  - [*getkey()*](https://docs.python.org/3/library/curses.html#curses.window.getkey): get a character and return a string instead of an integer.
  - [*getstr()*](https://docs.python.org/3/library/curses.html#curses.window.getstr): return a bytes object from the user.
- Erase (clear):
  - [*erase()*](https://docs.python.org/3/library/curses.html#curses.window.erase): clear the window but not updat until invoking [*refresh()*](https://docs.python.org/3/library/curses.html#curses.window.refresh).
  - [*clear()*](https://docs.python.org/3/library/curses.html#curses.window.clear): clear the window and update right away.
  - [*deleteln()*](https://docs.python.org/3/library/curses.html#curses.window.deleteln): delete the line under the cursor. All following lines are moved up by one.
  - [*clrtoeol()*](https://docs.python.org/3/library/curses.html#curses.window.clrtoeol): erase from cursor to the end of the line.
- Get from the window:
  - [*inch()*](https://docs.python.org/3/library/curses.html#curses.window.inch): return the character at the given position (use lower 8 bits), the upper bits are the attributes.
  - [*instr()*](https://docs.python.org/3/library/curses.html#curses.window.insstr): return a byts object extracted from the window; attributes are stripped from the characters.
- Update windows: the displayed content is not updated until explicit calls.
  - [*refresh()*](https://docs.python.org/3/library/curses.html#curses.window.refresh): for a window, update the display immediately. (This method internally calls *noutrefresh()* and then *doupdate()*.)
  - [*noutrefresh()*](https://docs.python.org/3/library/curses.html#curses.window.noutrefresh): for a window, update the underlying data structure but do not force an update aof the physical screen.
  - [*curses.doupdate()*](https://docs.python.org/3/library/curses.html#curses.doupdate): update the physical screen.
  - [Remark](https://docs.python.org/3/library/curses.html#curses.doupdate): If you have to update multiple windows, you can speed performance and perhaps reduce screen flicker by issuing *noutrefresh()* calls on all windows, followed by a single *doupdate()*.

Here is a small tip about the curses-coordinate philosophy: the whole module uses (y, x) as the coordinate, but this actually means the (row, col) position of a cell, and makes you consider a window as a matrix of shape (LINES, COLS).

You may want to try out this fun interactive program below (written by: me). It's a little verbose because the APIs are primitive. But as you can see, these simple APIs can be composed into wonderful things with your brilliant mind. 

```python
import curses
from curses import wrapper

class LineAdder:
    def __init__(self, win, start_line, *attr):
        self.win = win
        self.line = start_line
        self.attr = attr

    def add(self, message=""):
        self.win.addstr(self.line, 0, message, *self.attr)
        self.line += 1

    def skip(self, nlines):
        self.line += nlines

    def hline(self):
        self.win.hline(self.line, 0, "-", self.win.getmaxyx()[1])
        self.line += 1

    def replace(self, line_n, message=""):
        '''
        replace a given line with message, overwrite the previous characters
        if line_n < 0, then line_n is calculated as self.line + line_n
        '''
        if line_n <= 0:
            self.line += line_n
        else:
            self.line = line_n
        
        self.win.move(self.line, 0) # move to the given line
        self.win.addstr(message, *self.attr) # print message
        self.win.clrtoeol() # then clear to eol if the previous message is longer

        if message != "": # if message is not empty
            self.line += 1 # then self.line is added by 1 to keep consistency


def main(stdscr):
    stdscr.clear()
    
    # display a welcome banner
    stdscr.addstr("** HELLO, CURSES! **", curses.A_REVERSE)
    stdscr.chgat(-1, curses.A_REVERSE)

    # initialze some colors
    #curses.start_color() # you don't need this because the wrapper did it for you
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_BLUE, curses.COLOR_BLACK)

    #stdscr.getkey()
    stdscr.addstr(2, 0, "Curses module is really helpful!", curses.color_pair(1))
    
    # I implemented a simple class to add multiple lines in the same form
    line_adder = LineAdder(stdscr, 3, curses.color_pair(2))
    line_adder.add("Now let me show you how this works.")
    line_adder.hline()
    line_adder.add("Tell me your name, and I'll use getstr() to get it.")
    line_adder.add("Your name: ")

    curses.echo()
    name = stdscr.getstr(20).decode() # getstr() returns a bytes object, convert to str
    curses.noecho()

    line_adder.skip(1)
    line_adder.add(f"Now I know you are {name}! Hello, {name}!")
    line_adder.skip(1)

    line_adder.add("I guess you are a girl! Am I right? [y / n] ")

    # a basic question-based dialog loop
    while True:
        
        curses.echo()
        respon = stdscr.getkey().lower()
        curses.noecho()
        line_adder.replace
        if respon == "y":
            line_adder.skip(1)
            line_adder.add("Yah, so lucky!")
            break

        elif respon == "n":
            line_adder.add("Sorry, pls forgive me. I'm just a poor program who knows nothing ...")
            stdscr.getkey()
            line_adder.replace(-1)
            line_adder.replace(-1, "Then you must be a boy! [y / n]")
        
        else:
            line_adder.add("Invalid answer, use 'y' or 'n': ")
            
    #stdscr.refresh()
    stdscr.getkey() # block the screen from terminating
```

Reference: [Python HOWTOs](https://docs.python.org/3/howto/curses.html); [Python curses docs](https://docs.python.org/3/library/curses.html).

For those who want to achieve more in terminal displays, a good suggestion for your next step is the [Urwid](http://urwid.org/tutorial/index.html) module, which provides a higher-level abstraction, supporting common UI widgts such as buttons, textboxes, charts, diagrams and so forth. It may be a more difficult topic, but once you conquer it, it becomes your powerful tool.

# Json | JSON Encoder and Decoder

```python
import json
```

First and foremost, why and when should we use json objects instead Python original data types.

The [json](https://docs.python.org/3/library/json.html) complies with [RFC 7159](https://tools.ietf.org/html/rfc7159.html) in most aspects with minor exceptions. It exposes a user-friendly API to handle json objects.

To load (serialize) a json object from a formatted string, use [*json.loads()*](https://docs.python.org/3/library/json.html#json.loads).

```python
>>> s = '["foo", {"bar":["baz", null, 1.0, 2]}]'
>>> json.loads(s)
['foo', {'bar': ['baz', None, 1.0, 2]}]
```

- When the formatted string has repeated names (which is not permitted in RFC), this module ignores all but the last name-value pair for a given name.

  ```python
  >>> weird_json = '{"x": 1, "x": 2, "x": 3}'
  >>> json.loads(weird_json)
  {'x': 3}
  ```

To do the opposite (encode a *dict* or *list* into a json formatter string), use [*json.dumps()*](https://docs.python.org/3/library/json.html#json.dumps).

```python
>>> json.dumps({1: ["alice", "bob"], "relation": "lovers"})
'{"1": ["alice", "bob"], "relation": "lovers"}'
>>> json.dumps([True, None, float("inf")])
'[true, null, Infinity]'
```

- Non-string keys are all converted into strings. So `loads(dumps(x)) != x` if x has non-string keys.
-  Values are converted to corresponding json formats:
  - `True/False -> true/false`
  - `None -> null`
  - `float('nan'/'+inf'/'-inf') -> NaN/Infinity/-Infinity`
  - The last two set of values are not permitted by RFC. However, this module considers them valid JSON number literal values. Set argument *allow_nan* as False to enforce a strict compliance with RFC.

Another pair of decode / encode methods, [*json.load()*](https://docs.python.org/3/library/json.html#json.load) and [*json.dump()*](https://docs.python.org/3/library/json.html#json.dump), share most functionalities with the two methods above, but are designed for IO streams, e.g. file streams.

```python
# Writing JSON data
with open('data.json', 'w') as f:
    json.dump(data, f)

# Reading data back
with open('data.json', 'r') as f:
    data = json.load(f)
```

More advanced usage of the json module, including details about *load()* and *dump()* methods, serializing objects that are not standards JSON types, and playing with JSONEncoder and JSONDecoder, can be found in the following two links.

References:

- [Python3 cookbook: 读写JSON数据](https://python3-cookbook.readthedocs.io/zh_CN/latest/c06/p02_read-write_json_data.html)
- [Python documentation: json -- JSON encoder and decoder](https://docs.python.org/3/library/json.html)

# Bytes and Bytearray

Bytes and bytearray are two new types introduced in Python 3.x. They represent a sequence of 8-bit integers, or bytes. In fact, a byte sequence is exactly what resides in the memory, without any additional information about its meaning. If properly explained, a byte sequence can be a string, an image, or a binary file.

Let's start with bytes. As said before, a bytes object is a sequence, and each element is an 8-bit integer. So, you can create a bytes object from a list of integers within the range 0 to 255.

```python
>>> obj = bytes([1, 2, 3])
>>> obj
b'\x01\x02\x03'
```

`obj` is a bytes object. It is displayed as `b'\x01\x02\x03`. This means this object contains three bytes: 0x01, 0x02, and 0x03.

By default, these values are interpreted as **ASCII** characters. In ASCII, integer 0 is 0x30 (48), character A is 0x41 (65), and character a is 0x61 (97). Now that we know a bytes object can be interpreted as a ASCII sequence, let's spell the sentence "Hello, Bytes!" in ASCII.

```python
>>> bytes([72, 101, 108, 108, 111, 44, 32, 66, 121, 116, 101, 115, 33])
b'Hello, Bytes!
```

Amazing! Isn't it? But, if we always have to create a bytes object in this cumbersome style, then we won't even think of using it! Luckily, creating bytes objects from *int* lists is introduced only to give you an insight into bytes. Now let's look at other convenient ways to create a bytes object.

- *bytes(string, coding)*: encodes a string to a byte sequence (must specify the *coding* method). (refer to this [Chinese post](https://zhuanlan.zhihu.com/p/46216008) to learn more about coding methods and their relationships)

  ```python
  >>> bytes("message", encoding="ascii")
  b'message'
  >>> bytes("信息", encoding="utf-8")specifying the coding method, str.encode() uses ascii by default.
  b'\xe4\xbf\xa1\xe6\x81\xaf'
  >>> bytes("信息", encoding="gbk")
  b'\xd0\xc5\xcf\xa2'
  ```

- *b*-prefix strings: equivalent to *bytes(string, encoding="ascii")*.

  ```python
  >>> b"interesting"
  b'interesting'
  >>> b"信息"
    File "<stdin>", line 1
  SyntaxError: bytes can only contain ASCII literal characters.
  ```

- *bytes(n_bytes)*: returns a byte sequence of length *n_bytes*, filled with 0x00.

```python
>>> bytes(5)
b'\x00\x00\x00\x00\x00'
>>> bytes() # or bytes(0)
b''
```

- *str.encode(coding)*: same as *bytes(string, coding)*, with one difference. Without specifying the *coding* method, *str.encode()* uses ascii by default.

To convert a bytes object to a string, call *decode()*.

```python
>>> b'\xe4\xbf\xa1\xe6\x81\xaf'.decode("utf-8")
'信息'
```

I guess you have grasped the philosophy of bytes. Now let's talk a little more about *bytearray*. *Bytearray* can be considered as an extension of *bytes*. Bytearray objects allow modification to the byte sequence, including *append()*, *pop()*, *extend()*, *insert()*, *remove()*, *find()*, *replace()* and other methods that also appears in the *list* type. 

```python
>>> obj = bytearray("Hello", encoding="utf-8")
>>> # append: must be an integer, can't use bytes objects
>>> obj.append(44) # 44 = ,
>>> obj.append(32) # 32 = (space)
>>> obj
bytearray(b'Hello, ')
>>> # extend
>>> obj.extend(b"Bytearray!")
>>> obj
bytearray(b'Hello, Bytearray!')
>>> # replace: copy and replace
>>> obj.replace(b'l', b'y')
bytearray(b'Heyyo, Bytearray!')
>>> # reverse: in-place
>>> obj.reverse()
>>> obj
bytearray(b'!yarraetyB ,olleH')
```

Remark that bytes objects do not support all of these methods, because a byte object can not be modified after creation.

# 2to3 | Automated Python 2 to 3 code translation

This is command-line Python program that translates Python 2 codes to Python 3. For a customized usage, see [lib2to3](https://docs.python.org/3.7/library/2to3.html#module-lib2to3).

Say we have a Python 2.x source file, named `example.py`.

```python
def greet(name):
    print "Hello, {0}!".format(name)
print "What's your name?"
name = raw_input()
greet(name)
```

`$ 2to3 example.py` shows you the difference between Python 2 and 3 codes after translation. But it won't really do the translation.

To write translated codes into the source file, use `-w` argument.

```bash
$ 2to3 -w example.py
```

This will write the translated codes into `example.py` and create a backup file named `example.py.bak`.

After translation, `example.py` looks like this:

```python
def greet(name):
    print("Hello, {0}!".format(name))
print("What's your name?")
name = input()
greet(name)
```



