---
layout:     post
title:      C++ | Data Structures in Standard Library, a Summary
subtitle:   a summary of standard C++ data structures
date:       2020-01-16
author:     Xinyu Wang
header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - C++
    - Data structures
---

Suggestions to myself on writing posts.

- Try using bullet points to explain an idea.
- Always remember that copying official documents does nothing more than wasting time. The only thing to help memorizing is to read and then practice.

Data structures in standard library:

- [std::*array*](http://www.cplusplus.com/reference/array/) is a fixed-size sequence container. It holds elements in a strict linear order.
	```c++
	#include <array>
	template < class T, size_t N > class array;
	```
- [std::*deque*](http://www.cplusplus.com/reference/deque/deque/)
- [std::*map*](http://www.cplusplus.com/reference/map/map/) is a binary search tree which holds a key and its mapped value in each node.
	- Compared with std::*unordered_map*, it is slower to access individual elements by their keys ( *O(log n)* ), but it allows the direct iteration on subsets based on their order. (I'm still confused about the benefits of ordering keys.)
	- Compared with std::*multimap*, it does not allow multiple elements to hold an equivalent key. The key of every element is unique.
	```c++
	#include <map>
	template < class Key,									// map::key_type
	           class T,										// map::mapped_type
			   class Compare = less<Key>					// map::key_compare
			   class Alloc = allocator<pair<const Key, T> >	// map::allocator_type
			   > class map;
	```
- [std::*multimap*](http://www.cplusplus.com/reference/map/multimap/) is a binary search tree which holds a key and its mapped value in each node, and allows multiple elements to hold an equivalent key.
	```c++
	#include <map>
	template < class Key,									// map::key_type
	           class T,										// map::mapped_type
			   class Compare = less<Key>					// map::key_compare
			   class Alloc = allocator<pair<const Key, T> >	// map::allocator_type
			   > class multimap;
	```	
- [std::*unordered_map*](http://www.cplusplus.com/reference/unordered_map/unordered_map/) is a hash table.
	- Compared with std::*map*, it is much faster to access individual elements by their keys ( O(1) ).
	- Compared with std::*unordered_multimap*, it does not allow multiple elements to hold an equivalent key. The key of every element is unique.
	```c++
	#include <unordered_map>
	template < class Key,										// unordered_map::key_type
	           class T,											// unordered_map::mapped_type
			   class Hash = hash<Key>,							// unordered_map::hasher
			   class Pred = equal_to<Key>,						// unordered_map::key_equal
			   class Alloc = allocator< pair<const Key, T> >	// unordered_map::allocator_type
			   > class unordered_map;
	```
- [std::*unordered_multimap*](http://www.cplusplus.com/reference/unordered_map/unordered_multimap/)
	```c++
	template < class Key,                                   // unordered_multimap::key_type
			   class T,										// unordered_multimap::mapped_type
               class Hash = hash<Key>,                      // unordered_multimap::hasher
               class Pred = equal_to<Key>,                  // unordered_multimap::key_equal
               class Alloc = allocator< pair<const Key,T> >	// unordered_multimap::allocator_type
               > class unordered_multimap;
	```
- [std::*vector*](http://www.cplusplus.com/reference/vector/vector/) is an array that dynamically changes in size to fit with its elements.
	```c++
	#include <vector>
	template < class T, class Alloc = allocator<T> > class vector;
	```

