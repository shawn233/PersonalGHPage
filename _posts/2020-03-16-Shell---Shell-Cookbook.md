---
layout:     post
title:      Shell | Shell Cookbook
subtitle:   how to write a shell script
date:       2020-03-16
author:     Xinyu Wang
header-img: img/post-bg-cook.jpg
catalog:    true
tags:
    - Shell
    - Linux
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

# Quick Starter

This section is intended for experienced programmers who may have a long history working with C++, Java, or Python, but have no idea how to write a shell script. For these readers, I try to offer them a quick start in shell by bridging shell concepts to those they are already familiar with in other languages.

For me, **shell** is a powerful *string* processing program integrated with complete *logical controls* and *internal system functionalities*.

By writing a shell script, it may perform the following tasks:

- operating efficiently on *files* and *directories*
- executing and combining *shell commands* for more complex processing
- interacting with *command-line* users in a simple and nice way

Ensure you do have basic background knowledge about those concepts I marked with *Italian font*. Now let's dive into the wondeful shell!

## Defining variables

One reason to write a script instead of running commands is that we want earlier execution results to influence later execution process. For example, we want to make sure a directory exists. If it doesn't, we should create it before we do other things in it. In Python this task can be performed by

```python
target_dir = "./foo"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
# now we are sure ./foo exists
```

To do this in a shell script, we first need a variable to hold the path like *target_dir* did in Python. There are mainly (how many) ways to define a variable.

### Direct assignment

Firstly, we can directly assign values to a variable.

```bash
target_dir="./foo"
target_dir='./foo'
target_dir=./foo
```

- There must be no spaces on both sides of `=`, unlike other languages. It is illegal to assgin values with `var = something`.

- Use quotation marks (`" "` or `' '`) when a string itself contains spaces, such as `"hello world"`.

- `${variable}` or `$variable` uses the value of `variable`.

- Single quotation marks (`' '`) does not replace variables with their values. Printing such strings yields the original content. On the other hand, double quotation marks (`" "`) are much smarter since it recognizes variables and replace them with their values. 

  ```bash
  target_dir="./foo"
  target_filename="boo.txt"
  
  target_path='${target_dir}/${target_filename}'
  echo $target_path # original content
  # output: ${target_dir}/${target_filename}
  
  target_path="${target_dir}/${target_filename}"
  echo $target_path # replace variables
  # output: ./foo/boo.txt
  ```

-  use `echo -e` to translate escape characters such as `\n` and `\t`.

  ```bash
  hi='hello\nbash'
  echo $hi
  # output: hello\nbash
  echo -e $hi # -e argument translates escape characters
  # output: hello
  #       : bash
  ```

All assignments above work. But as an experienced programmer, you may want to know more about the internals of the direct assignment. As I've mentioned,

> For me, **shell** is a powerful string processing program

By using the term "string processing program", I literally mean it only processes **strings**. See the code above, we assign a string to a variable, but we do not need to wrap the strings with `" "` or `' '` like other languages. Thus, even when we want to assign integer values to a variable, shell recognizes those digits as strings.

```bash
# shell thinks every value assignment as string assigment
a=123 # no difference to a="123"
b=hello123
```

- Although shell considers all values as strings, it is able to recognize strings that are essentially numbers, and even provides such condition statements to determine whether a string is a number.

### From execution

Secondly, we can have variables to hold the results of command execution.

```bash
PWD=`pwd`
PWD=$(PWD)
echo $PWD
# output: /home/shawn233
val=`expr 1 + 2` 
echo $val
# output: 3
val=$((1 + 2))
echo $val
# output: 3
```

- `` ` ` `` and `$( )` support running any command and assign the result to the variable.

- `$(( ))` is an abbreviation of `` `expr ` ``.

- *expr* is a command-line program which evaluates numerical expressions. It supports common operators such as +, -, \*, /, %, &, \|, <, <=, >, >=, regular expressions, string matching, and string indexing. In a shell environment, we must use spaces to separate each term. Also, some operators need to be escaped such as `\*`, `\<`, and `\( \)`. See `man expr` for more info.

  ```bash
  expr 1 + 2       # = 3
  expr 5 \* 10     # = 50 (needs to escape * operator)
  expr 50 / 10 / 2 # = 2 (expr only supports int)
  expr 50 / \( 10 / 2 \)     # = 10 (escapes brackets)
  expr length "hello expr"   # = 10 (no surprise)
  expr substr "hello expr" 2 3 # = ell (indexes from 1)
  expr index "hello expr" eo # = 2 (the first occurence)
  ```

### Implicit assignment

### Special variables

## Logical controls

# Reference



