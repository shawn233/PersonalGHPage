---
layout:     post
title:      Python | 浅谈Python多线程
subtitle:   the use of threading package
date:       2018-07-19
author:     shawn233
header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - Python
---



## 1. 什么是多线程？

[百度百科](https://baike.baidu.com/item/%E5%A4%9A%E7%BA%BF%E7%A8%8B/1190404?fr=aladdin)：多线程（英语：multithreading），是指从软件或者硬件上实现多个线程并发执行的技术。具有多线程能力的计算机因有硬件支持而能够在同一时间执行多于一个线程，进而提升整体处理性能。

Python 多线程就是指从软件层面实现多个线程并发执行。举个简单的例子，几乎所有的图形界面软件，图形界面占一个线程，后端程序占用单独的线程，这样在后端程序运行时，前端的界面不会卡死。

在当前的Python版本（3.6）下，实现多线程开发主要依赖两个库：

* `import threading`
* `import queue`

下面就来谈谈怎么用这两个库，实现简单的多线程。

## 2. 怎么多线程？

最重要的，就是`threading.Thread`类。这个类在多线程开发时是怎样工作的？官方文档作出如下说明：

> [官方文档](https://docs.python.org/3.6/library/threading.html)：Once a thread object is created, its activity must be started by calling the thread’s [`start()`](https://docs.python.org/3.6/library/threading.html#threading.Thread.start "threading.Thread.start")method. This invokes the [`run()`](https://docs.python.org/3.6/library/threading.html#threading.Thread.run "threading.Thread.run") method in a separate thread of control.
>
> Once the thread’s activity is started, the thread is considered ‘alive’. It stops being alive when its [`run()`](https://docs.python.org/3.6/library/threading.html#threading.Thread.run "threading.Thread.run") method terminates – either normally, or by raising an unhandled exception. The [`is_alive()`](https://docs.python.org/3.6/library/threading.html#threading.Thread.is_alive "threading.Thread.is_alive") method tests whether the thread is alive.

文档说，我们把`Thread`对象实例化后，调用它的`start()`函数，这个对象就开始工作了，这时这个对象所在的线程被视为 "alive" 。`start()`函数会自动调用`run()`函数，`run()`函数中就是线程主要做的事情。当线程的事情做完，`run()`函数结束时，线程停止，此时线程被视作 "not alive"。

这样一来，我们就知道了`Thread`类的工作原理。那么我们应该怎样利用这个类，来让线程做我们希望的工作呢？

> [官方文档](https://docs.python.org/3.6/library/threading.html)：The [`Thread`](https://docs.python.org/3.6/library/threading.html#threading.Thread "threading.Thread") class represents an activity that is run in a separate thread of control. There are two ways to specify the activity: by passing a callable object to the constructor, or by overriding the [`run()`](https://docs.python.org/3.6/library/threading.html#threading.Thread.run "threading.Thread.run") method in a subclass. No other methods (except for the constructor) should be overridden in a subclass. In other words, *only* override the `__init__()` and [`run()`](https://docs.python.org/3.6/library/threading.html#threading.Thread.run "threading.Thread.run")methods of this class.

文档中提到，指定某一个特定线程的行为（就是这个线程执行什么指令）有两个途径：

* 将行为封装为一个可以调用的函数，然后传入`Thread`类的构造函数
* 继承`Thread`类，并定义自己的`__init__()`函数和`run()`函数

具体的，这两个途径的实现与`Thread`类的构造函数有关：

```
class threading.Thread(group=None, 
                       target=None, 
                       name=None, 
                       args=(), 
                       kwargs={}, 
                       *, 
                       daemon=None)
```

重要参数：

* `target`参数指定在`run()`中被调用的函数
* `name`指定了这个对象的名字
* `kwargs`为target函数指定传入的参数，形式为`dict`

关于`kwargs`传参，举个简单的例子

```
def work(n):
    print (n + 2)

t1 = threading.Thread(target=work, kwargs={'n':4})
t1.start()

# output: 6
```

现在，我们已经可以用第一个途径（将行为封装为一个可以调用的函数，然后传入`Thread`类的构造函数）来实现Python多线程。我们把线程的行为封装进函数，然后将函数对象传入`target`参数，把函数的参数通过`kwargs`传递，就可以使`Thread`对象做我们希望的工作。

当然，我们还可以有另一条途径，即继承`Thread`类，并定义自己的`__init__()`函数和`run()`函数。这条途径的原理在现在看来也很显然了：定义我们自己的`run()`函数，让`Thread`对象执行。这里需要注意一点，自定义的`__init__`函数，必须要调用`Thread`类的`__init__`函数。

```
def __init__ (self):
    threading.Thread.__init__(self)
```

其实，在自定义线程行为这件事情上，我们只要把握住一点：`start()`函数会自动调用`run()`函数。只要明白了`run()`函数在`Thread`类的作用，我们在使用`Thread`类时也会得心应手。

有了以上知识，我们在Python多线程开发上已经算入门了。当然，这些知识远远无法应付实际的需求，因为线程间并非毫无联系。多个线程并发执行时，资源的调配、数据的管理都是大问题。下面的内容将会对Python多线程作更深入的讲解。当然，内容主要基于[官方文档](https://docs.python.org/3.6/library/threading.html)，因此本文的知识实际上都可以通过阅读文档获得。

## 3. `join()`函数

## 4. `Lock`对象

## 5. `Daemon`属性

## 6. `queue`模块
