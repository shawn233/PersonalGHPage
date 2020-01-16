---
layout:		post
title:		C++ | std::vector, a Summary
subtitle:	a comprehensive note for std::vector
date:		2020-01-16
author:		Xinyu Wang
header-img:	img/post-bg-cook.jpg
catalog:	true
tags:
	- C/C++
---

This is my first summary of a standard C++ class template. As a starter of a very long series, I try to achieve the following goals with this post:

- Figure out a clear structure to summarize a class template, which I will employ in my later summary posts.
- Give you a concise impression of how std::vector is used with easy-to-interpret programming samples.
- Provide a comprehensive summary of scenarios where we frequently use std::vector and how we use it.

My summary has several parts.

- Introduction: to tell you what a std::vector is and how to use it in the simplest way.
- Class Template: a necessary part to tell you the template parameter of std::vector
- Scenarios: a summary of scenarios where we frequently use std::vector and how we use it.
  - Create a Vector: ways to create a vector `v`
  - Insert an element: ways to insert elements into vector `v`, featuring an analysis of time cost
  - Modify the vector: ways to modify vector `v`

Suggestions to myself on writing posts.

- Try using bullet points to explain an idea.

## Introduction

Vectors are sequence containers representing arrays that can change in size.

Vectors are very similar to arrays.

- Contiguous storage locations
- Efficient element access by using memory offsets

Vectors are different from arrays in that vector size can change dynamically.

- Better adaptability to elements with unknown amount
- Higher time and memory costs when expand vector size

Vector size is changed if and only if after insertion the new vector size surpasses the current vector capacity.
Internally, dynamic vector size may be implemented internally by:

- reallocating a larger contiguous memeory space and moving elements to the new storage
- allocating extra memeory space when vectors are full

Here is a simple example to use `std::vectors`.

> TODO: a simple example of using vector

## Class Template

```c++
template < class T, class Alloc = allocator<T> > class vector;
```

- T: type of the elements. (If T is move constructible, i.e., `is_nothrow_move_constructible.value == true`, then implementations can optimize to move elements instead of copying them during reallocations.) Aliased as (e.g. using `typedef`) member type `vector::value_type`
- Alloc: type of allocator object used to define the storage allocation model. The default value `allocator<T>` define the simplest memory allocation model and is value-independent. Aliased as member type `vector::allocator_type`

Member types:

![](https://github.com/shawn233/shawn233.github.io/raw/master/_posts/.assets/image-20200116111932792.png)

## Scenarios

### Create a vector

Official doc provides 6 ways to construct (create) a vector.

1. **empty container constructor (default constructor)**
	Constructs an empty container, with no elements.
	```c++
	explicit vector (const allocator_type& alloc = allocator_type());
	```

2. **fill constructor**
	Constructs a container with *n* elements. Each element is a copy of *val* (if provided)
	```c++
	explicit vector (size_type n);
	         vector (size_type n, const value_type& val,
					 const allocator_type& alloc = allocator_type());
	```

3. **range constructor**
	Constructs a container with as many elements as the range [*first*, *last*), with each element emplace-constructed from its corresponding element in that range, in the same order.
	```c++
	template <class InputIterator>
	vector (InputIterator first, InputIterator last,
			const allocator_type& alloc = allocator_type());
	```

4. **copy constructor**
	Constructs a container with a copy of each of the elements in *x*, in the same order.
	```c++
	vector(const vector& x);
	vector(const vector& x, const allocator_type& alloc);
	```

5. **move constructor**
	Constructs a container that acquires the elements of *x*.
	*x* is left in an unspecified but valid state. (??)
	```c++
	vector (vector&& x);
	vector(vector && x, const allocator_type alloc);
	```

6. **initilizer list constructor**
	Constructs a container with a copy of each of the elements in *il*, in the same order.
	```c++
	vector (initializer_list<value_type> il,
			const allocator_type& alloc = allocator_type());
	```

Sample code

```c++
// constructing vectors
#include <iostream>
#include <vector>

int main ()
{
  // constructors used in the same order as described above:
  std::vector<int> first;                                // empty vector of ints
  std::vector<int> second (4,100);                       // four ints with value 100
  std::vector<int> third (second.begin(),second.end());  // iterating through second
  std::vector<int> fourth (third);                       // a copy of third

  // the iterator constructor can also be used to construct from arrays:
  int myints[] = {16,2,77,29};
  std::vector<int> fifth (myints, myints + sizeof(myints) / sizeof(int) );

  std::cout << "The contents of fifth are:";
  for (std::vector<int>::iterator it = fifth.begin(); it != fifth.end(); ++it)
    std::cout << ' ' << *it;
  std::cout << '\n';

  return 0;
}
```

Output

```
The contents of fifth are: 16 2 77 29
```

### Insert an element

1. add an element at the end
	```c++
	void push_back (const value_type& val);
	void push_back (value_type &&val);
	```

2. insert elements
	```c++
	// (1) single element
	iterator insert (const_iterator position, const value_type& val);

	// (2) fill
	iterator insert (const_iterator position, size_type n, const value_type& val);
	
	// (3) range
	template <class InputIterator>
	iterator insert (const_iterator position, InputIterator first, InputIterator last);
	
	// (4) move
	iterator insert (const_iterator position, value_type&& val);

	// (5) initializer list
	iterator insert (const_iterator position, initializer_list<value_type> il);
	```

Sample code

```c++
// inserting into a vector
#include <iostream>
#include <vector>

int main ()
{
  std::vector<int> myvector (3,100);
  std::vector<int>::iterator it;

  it = myvector.begin();
  it = myvector.insert ( it , 200 );

  myvector.insert (it,2,300);

  // "it" no longer valid, get a new one:
  it = myvector.begin();

  std::vector<int> anothervector (2,400);
  myvector.insert (it+2,anothervector.begin(),anothervector.end());

  int myarray [] = { 501,502,503 };
  myvector.insert (myvector.begin(), myarray, myarray+3);

  std::cout << "myvector contains:";
  for (it=myvector.begin(); it<myvector.end(); it++)
    std::cout << ' ' << *it;
  std::cout << '\n';

  return 0;
}
```

Output

```
myvector contains: 501 502 503 300 300 400 400 200 100 100 100
```

### Modify the vector


