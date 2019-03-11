---
layout:     post
title:      Artificial Intelligence | Search
subtitle:   Python Implementation of Search Algorithms
date:       2018-09-27
author:     shawn233
header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - AI
    - Python
---

At the beginning, we should know the TREE-SEARCH and GRAPH-SEARCH principals that we will follow.

![](https://upload-images.jianshu.io/upload_images/10549717-8601f7cfaca6fcec.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/10549717-c6c69b9c5148163c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Also we should know some evaluation factors for various algorithms.

![](https://upload-images.jianshu.io/upload_images/10549717-a5e81087777ae470.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Before we begin introduction and simple implementation of different search algorithms, we define a structure to unite all the implementations given below, using Python, the powerful programming language.

```python
# encoding: utf-8
# file: search_algorithm.py
# author: shawn233
# start date: 2018-09-27

from __future__ import print_function

class Problem:

    def __init__ (self):
        pass
    
    def initialState (self):
        pass

    def action (self, node):
        pass
    
    def goalTest (self, node):
        pass

    def pathCost (self, node):
        pass

    def stepCost (self, node_from, node_to):
        pass

class SearchAlgorithm:

    def __init__ (self, problem):
        self.problem = problem

    def expand (self, node):
        return self.problem.action (node)
```

The first several algorithms belong to uninformed search. They are relatively trivial, so I just give necessary points to introduce these algorithms.

### Breadth-First Search

* Strategy: Expand shallowest unexpanded node
* Advantage: find the path of minimal length to the goal
* Disadvantage: require the generation and storage of a tree whose size is exponential the depth of the shallowest goal node
* Evaluation:
  * Completeness: Yes if b is finite
  * Time complexity: O(b<sup>d+1</sup>)
  * Space complexity: O(b<sup>d+1</sup>)
  * Optimality: Yes if step cost is a constant
* Implementation: Use FIFO queue

```python
class SearchAlforithm:

    # ...

    def breadthFirstSearch (self):
        q = queue.Queue()
        q.put (self.problem.initialState())
        while True:
            if q.empty():
                print ("[warning] no solution.")
                return
            state = q.get()
            if self.problem.goalTest(state):
                print ("Solution found!")
                return state
            successors = self.problem.action (state)
            for sc in successors:
                q.put (sc)
```

### Uniform-Cost Search

* Strategy: expand the least -cost unexpanded node
* Evaluation:
  * Completeness: Yes if step cost is greater than ε
  * Time complexity: number of nodes with cost less than the cost of optimal solution, i.e. O(b<sup>ceiling(C*/ε)</sup>), where C* is the cost of the optimal solution
  * Space complexity: number of nodes with cost less than C*, i.e. O(b<sup>ceiling(C*/ε)</sup>)
  * Optimality: Yes for nodes expand in ascending order of cost.
* Implementation key idea: make `fringe` a priority queue, ordered by the path cost
* Implementation

```python
class ComparableNode:
    '''
    Encapsulate nodes in this class to support a priority queue
    '''

    def __init__ (self, node, measure):
        self.node = node
        self.measure = measure

    def __cmp__ (self, other):
        if self.measure < other.measure:
            return True
        else:
            return False

    def getNode (self):
        return self.node

    def getMeasure (self):
        return self.measure

class SearchAlgorithm:

    # ...

    def uniformCostSearch (self):
        q = queue.PriorityQueue()
        q.put (ComparableNode(self.problem.initialState(), 0))
        current_cost = 0
        while True:
            if q.empty():
                print ("[warning] no solution.")
                return
            comparable_node = q.get() # of type ComparableNode
            state = comparable_node.getNode()
            current_cost = comparable_node.getMeasure()
            if self.problem.goalTest (state):
                print ("Solution found!")
                return state
            successors = self.problem.action (state)
            for sc in successors:
                cost = current_cost + self.problem.stepCost (state, sc)
                q.put (ComparableNode(sc, cost))
```

### Depth-First Search

* Strategy: Expand the deepest unexpanded node.
* Evaluation:
  * Completeness: No if space with depth of infinity occurs
  * Time complexity: O(b<sup>m</sup>). Recall m is the maximum depth of the state space
  * Depth complexity: O(bm)
  * Optimality: No
* Implementation: Use LIFO queue

```python
class SearchAlgorithm:

    # ...

    def depthFirstSearch (self):
        q = queue.LifoQueue()
        q.put (self.problem.initialState())
        while True:
            if q.empty():
                print ("[warning] no solution.")
                return
            state = q.get()
            if self.problem.goalTest(state):
                print ("Solution found!")
                return state
            successors = self.problem.action (state)
            for sc in successors:
                q.put (sc)
```

### Depth-Limited Search

Because we find that the depth-first search algorithm fails when infinite depth exists in the state space, we manually set a limit for the search depth in order to prevent the search from exploring too deep into a branch.

* Strategy: Set a depth limit l for the depth-first search algorithm
* Evaluation:
  * Completeness: No
  * Time complexity: number of nodes visited within the limited depth d. N<sub>DLS</sub> = b<sup>0</sup> + b<sup>1</sup> + ... + b<sup>d</sup> = O(b<sup>d</sup>)
  * Space complexity: O(bd)
  * Optimality: No from DFS
* Implementation: There are two major ways of implementation, recursive or non-recursive. Considering the poor performance in function calls of Python, I present a non-recursive implementation.

```python
class SearchAlgorithm:

    # ...

    def depthLimitedSearch (self, l):
        '''
        Argument l indicates the depth limit
        '''
        q = queue.LifoQueue()
        depth_q = queue.LifoQueue()
        q.put (self.problem.initialState())
        depth_q.put (0)
        visit_cnt_q.put (0)
        while True:
            if q.empty ():
                print ("[warning] no solution.")
                return
            state = q.get()
            if self.problem.goalTest(state):
                print ("Solution found!")
                return state
            depth = depth_q.get()
            if depth == l:
                continue
            successors = self.problem.action (state)
            for sc in successors:
                q.put (sc)
                depth_q.put (depth + 1)
```

### Iterative Deepening Search

![](https://upload-images.jianshu.io/upload_images/10549717-94def53e32cb920d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

* Strategy: Iteratively search by depth-limited algorithm with depth limit increased
* Advantage:
  * Linear memory requirements acquired from depth-limited search
  * Guarantee for goal node of minimal depth
* Evaluation:
  * Completeness: Yes
  * Time /Space complexity: N<sub>IDS</sub> = (d+1)b<sup>0</sup> + db<sup>1</sup> + ... + 2b<sup>d-1</sup> + b<sup>d</sup> = O(b<sup>d</sup>)
  * Space complexity: O(bd)
  * Optimality: Yes if step cost is a constant

```python
class SearchAlgorithm:

    # ...

    def iterativeDeepeningSearch (self, max_depth = 1e5):
        for depth in range (max_depth):
            res = self.depthLimitedSearch(depth)
            if res:
                return res
        print ("[warning] iterative deepening search can not find a solution in depth range: <", max_depth)
```

Now we come to the topic of informed search.

### Best-First Search

A typical informed search algorithm is the best-first search algorithm. It introduces an **evaluation function**  f(n) to estimate our desire to expand each node. In each step, we should always **expand the most desirable unexpanded node**. A natural implementation is to store nodes in a priority queue sorted by the value of evaluation.

A best-first search is determined by the evaluation function, so in this part, I will define the evaluation function as an interface, and write a template of the best-first search. Later we will introduce two special cases in the best-first search, namely, the greedy best-first search and the A* search. We will implement them imitating this template.

```python
class SearchAlgorithm:

    # ...

    def bestFirstSearch (self, evaluation):
        '''
        Only a template, can not run
        '''
        q = queue.PriorityQueue()
        q.put (ComparableNode(self.problem.initialState(), 0))
        while True:
            if q.empty():
                print ("[warning] no solution.")
                return
            comparable_node = q.get()
            state = comparable_node.getNode()
            if self.problem.goalTest(state):
                print ("Solution found!")
                return state
            successors = self.problem.action (state)
            for sc in successors:
                q.put (ComparableNode(sc, evaluation(sc)))
```

### Best-First Search I - Greedy Best-First Search

Introduce another function: **heuristic function** h(n). This function returns an estimated cost from n to the goal. In this algorithm, we use heuristic function as evaluation function, i.e. f(n) = h(n)

In short, the strategy is to expand the node that appears to be the closest to the goal.

* Evaluation:
  * Completeness: No because the search may get stuck in some loop
  * Time complexity: O(b<sup>m</sup>)
  * Space complexity: O(b<sup>m</sup>) for all nodes are stored
  * Optimality: No
  

The implementation is actually based on the best-first search template defined above.

```python
class SearchAlgorithm:

    # ...

    def greedyBestFirstSearch(self, heuristic):
        return self.bestFirstSearch(heuristic)
```

### Best-First Search II - A* Search

The key idea of updating greedy bf search to A* search is to modify the evaluation function f(n) from h(n) to g(n) + h(n), where g(n) is the path cost from initial state to n. 

So in short the strategy is to avoid expanding nodes that are expensive.

* Evaluation:
  * Completeness: Yes
  * Time complexity: Exponential
  * Space complexity: All nodes are stored in memory
  * Optimality: Yes

```python
class AStarNode (ComparableNode):

    def __init__ (self, node, measure, path_cost):
        ComparableNode.__init__(node, measure)
        self.path_cost = path_cost
    
    def getPathCost (self):
        return self.path_cost

class SearchAlgorithm:

    # ...

    def aStarSearch (self, heuristic):
        q = queue.PriorityQueue()
        q.put (AStarNode(self.problem.initialState(), 0, 0))
        while True:
            if q.empty ():
                print ("[warning] no solution.")
                return
            a_star_node = q.get()
            state = a_star_node.getNode()
            if self.problem.goalTest (state):
                print ("Solution found!")
                return state
            path_cost = a_star_node.getPathCost()
            successors = self.problem.action (state)
            for sc in successors:
                pc = path_cost + self.problem.stepCost(state, sc)
                measure = heuristic(sc) + pc
                q.put (AStarNode(sc, measure, pc))
```

Now we come to another category of search algorithms: Local search algorithms.

### Hill-Climbing Search

The name of this algorithm means we should iteratively update the current state towards the optimal state.

The implementation differs in various cases, so here I only give an algorithm description.

![](https://upload-images.jianshu.io/upload_images/10549717-b2249c7b72bb498d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

The disadvantage of this algorithm is obvious: the search gets stuck in some local extremes.

### Simulated Annealing Search

This algorithm updates the hill-climbing algorithm, solving the problem of local extremes using some probability method.

The key idea is to allow some bad moves in the hill-climbing search in a decreasing probability.

The algorithm description is given below.

![](https://upload-images.jianshu.io/upload_images/10549717-505e9c8f88866a8a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### Local Beam Search

This algorithm is essentially running several hill-climbing searches simultaneously. It also aims at solving the problem of local extremes, just like the simulated annealing search. 

![](https://upload-images.jianshu.io/upload_images/10549717-f070009c75264edd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### Genetic Algorithm

This is big topic, learn this algorithm by googling!



