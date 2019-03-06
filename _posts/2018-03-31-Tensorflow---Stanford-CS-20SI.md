---
layout: post
title: Tensorflow | Stanford CS 20SI
date: 2018-03-31
author: shawn233
header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - Python
    - ML
---



## 1 Tensor

---

Tensor is basically **an n-dimensional matrix**.

* 0-d tensor: scalar ( `shape=()` )
* 1-d tensor: vector
* 2-d tensor: matrix

and so on.

### Session
---

> A Session object encapsulates the environment in which Operation objects are executed, and the Tensor objects are evaluated.

**1. Create a session**
```
sess = tf.Session()
```

**2. Run the graph**
```
 sess.run(<arg>)
```

In the parentheses, the argument could either be a graph, or a   node.

**3. Close a session**
```
sess.close()
```

**4. Another way to use a Session**

```
with tf.Sesstion() as sess:
    <commands>
```

In this case, you do not need an explicit `sess.close()`.

**5. About sess.run()**

```
tf.Session.run(fetches, feed_dict=None, options=None, run_metadata=None)
```

You could pass the nodes you want to calculate as a `list` to the argument `fetches`, and this function will return the result as a list.

For example,
```
x = 3
y = 2
op1 = tf.add(x, y)
op2 = tf = multiply(x, y)
useless = tf.multiply(op1, x)
op3 = tf.pow(op2, op1)
with tf.Session() as sess:
    op3, not_useless = sess.run([op3, not_uselesss])
```

**6. InteractiveSession**

In the terminal, you could use
```
tf. InteractiveSession()
```

to create an interactive environment, in which all ops will be run immediately. In this case, you could use the `<object>.eval()` to get the result immediately.

For example:
```
>>> tf.InteractiveSession()
2018-03-31 12:11:54.510828: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
<tensorflow.python.client.session.InteractiveSession object at 0x7f242ad01190>
>>> a = tf.constant([2, 2], name='a')
>>> b = tf.constant([[0, 1], [2, 3]], name='b')
>>> add = tf.add(a, b, name='add')
>>> add.eval()
array([[2, 3],
       [4, 5]], dtype=int32)
```

## 2 Graph
---

**1. Create a new graph and add operations**

```
g = tf.Graph()
with g.as_default():
    <commands>
sess = tf.Session(graph=g) # session is run on graph g
sess.run # run session
```

The `graph=g` indication could not be omitted, or the Session object would just run the default graph, in which there is no `op1`.

If you use `sess = tf.Session()` in the above codes:
```
RuntimeError: The Session graph is empty.  Add operations to the graph before calling run().
```

**2. Get the handle of the default graph**

```
g = tf.get_default_graph()
```

**3. Do not mix the default graph and the user created graph**

```
g = tf.Graph()

# add ops to the default graph
a = tf.constant(3)

# add ops to the user created graph
with g.as_default():
   b = tf.constant(5)
```

The codes above are equivalent to the following codes. And the following codes are better in some sense. However, having more than one graph is never recommended.

```
g1 = tf.get_default_graph()
g2 = tf.Graph()

# add ops to the default graph
with g1.as_default():
    a = tf.constant(3)

# add ops to the user created graph
with g2.as_default():
    b = tf.constant(5)
```

## 3 Visualize Graph Using Tensorboard
---

**1. Write down your events**

Create the summary writer after the definition of your graph and before running your session.

```
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    sess.run(x)
writer.close() # Close the writer when you are done using it
```

**2. Run your event files**

In terminal, run it.

```
$ python [yourprogram].py
$ tensorboard --logdir="./graphs" --port 6006
```

Then open your browser and go to: `http://localhost:6006/`.

## 4 Constant
---

**1. Signature**

```
tf.constant (value, dtype=None, shape=None, name='Const', verify_shape=False)
```

If you set the `verify_shape` argument as default (False), then the constant will fill the shape if the given value and shape are unmatched.

It's recommended to just give the value, and let the shape to be inferred.

**2. Tensor filled with a specific value -- 0**

```
tf.zeros(shape, dtype=tf.float32, name=None)

tf.zeros([2, 3], dtpye=tf.int32) ==> [[0, 0, 0], [0, 0, 0]]
```

```
tf.zeros_like(input_tensor, dtype=None, name=None, optimize=True)

# input tensor is [[0, 1], [2, 3], [4, 5]]
tf.zeros_like(input_tensor) ==> [[0, 0], [0, 0], [0, 0]]
```

**3. Tensor filled with a specific value -- 1**
```
tf.ones(shape, dtype=tf.float32, name=None)
tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)
```

**4. Tensor filled with a specific value -- value**
```
tf.fill(dims, value, name=None)

tf.fill([2, 3], 8) ==> [[8, 8, 8], [8, 8, 8]]
```

**5. Constants as sequences**

```
tf.linspace(start, stop, num, name=None) # start and stop should be of type float
tf.linspace(10.0, 13.0, 4) ==> [10.0 11.0 12.0 13.0]

tf.range(start, limit=None, delta=1, dtype=None, name='range') 
# The interval would be [start, limit), i.e., the limit is not included
tf.range(start=3, limit=18, delta=3) ==> [3, 6, 9, 12, 15]

tf.range(limit)
tf.range(5) ==> [0, 1, 2, 3, 4]
```

Also you should note that tensor objects are not iterable.

```
for _ in tf.range(4):
    pass
# TypeError
```

**6. Randomly Generated Constants**

```
#random_normal samples from a normal distribution
tf.random_normal (shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

#truncated_normal samples from a truncated normal distribution interval (mean-stddev, mean+stddev)
tf.truncated_normal (shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

tf.random_uniform (shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)

# shuffle only on the first dimension
tf.random_shuffle(value, seed=None, name=None)

# randomly crop some values with the given size (shape) among the given value
tf.random_crop(value, size, seed=None, name=None)

tf.multinomial(logits, num_samples, seed=None, name=None)

# sample from the gamma distribution with argument alpha
tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)
```

Set the seed

```
tf.set_random_seed(seed)
```

## 5 Operations
---

I think most tensorflow operations are element-wise.

```
# In the tf.InteractiveSession()

>>> a = tf.constant([3, 6])
>>> b = tf.constant([2, 2])

>>> a.eval()
array([3, 6], dtype=int32)
>>> b.eval()
array([2, 2], dtype=int32)

>>> tf.add(a, b).eval()
array([5, 8], dtype=int32)

>>> tf.add_n([a, b, b]).eval()
array([ 7, 10], dtype=int32)

>>> tf.multiply(a, b).eval()
array([ 6, 12], dtype=int32)

>>> tf.matmul(a, b)
ValueError: Shape must be rank 2 but is rank 1 for 'MatMul' (op: 'MatMul') with input shapes: [2], [2].
>>> tf.matmul(tf.reshape(a, [1, 2]), tf.reshape(b, [2, 1])).eval()
array([[18]], dtype=int32)

>>> tf.divide(a, b).eval()
array([1.5, 3. ])

>>> tf.mod(a, b).eval()
array([1, 0], dtype=int32)
```

## 6 Tensorflow Data Types
---

Tensorflow understands these basic types: `boolean`, numeric(`int`, `float`), `strings`

```
>>> t_0 = 19
>>> tf.zeros_like(t_0).eval()
0
>>> tf.ones_like(t_0).eval()
1

>>> t_1 = ['apple', 'peach', 'banana']
>>> tf.zeros_like(t_1).eval()
array(['', '', ''], dtype=object)
>>> tf.ones_like(t_1).eval()
TypeError: Expected string, got 1 of type 'int' instead.

>>> t_2  = [[True, False], [False, True]]
>>> tf.zeros_like(t_2).eval()
array([[False, False],
       [False, False]])
>>> tf.ones_like(t_2).eval()
array([[ True,  True],
       [ True,  True]])
```

Do not use Python native types.

Beware Numpy and Tensorflow might become not so compatible in the future.

## 7 Variables
---

**1. Create a variable**

```
# create variable a with scalar value
a = tf.Variable(2, name="scalar")

# create variable b as a vector
b = tf,Variable([2, 3], name="vector")

# create variable c as a 2x2 matrix
c = tf.Variable([[0, 1], [2, 3]], name="matrix")

# create variable W as 784 x 10 tensor, filles with zeros
W = tf.Variable(tf.zeros([784, 10]))
```

**2. Initialize your variables**

The easiest way is to initialize all variables at once.

```
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```

Initialize only a subset of variables

```
init_ab = tf.variables_initializer([a, b], name="init_ab")
with tf.Session() as sess:
    sess.run(init_ab)
```

Initialize a single variable

```
W = tf.Variable(tf.zeros([784, 10]))
with tf.Session() as sess:
    sess.run(W.initializer)
```

**3. Eval a variable**

Call the `eval()` op of a variable.

**4. Assign a variable**

```
W = tf.Variable(10)
W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    print W.eval() # >> 10
```

Note that the output of the codes above is 10, not 100. This is because that `W.assign(100)` doesn't assign the value 100 to W. It creates an assign op, and the op needs to be **`run`** to take effect.

 * So the correct version is:

```
W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    sess.run(assign_op)
    print W.eval() # >> 100
```

Note that `sess.run(W.initializer)` must be placed before `sess.run(assign_op)`, otherwise the output is still 10. I guess it's because the latter initialize op changes the value of W from 100 to 10. (Refer to the following part)

* Actually, assign op will initialize the variable, so you don't need the first initialize op.

```
W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    sess.run(assign_op)
    print W.eval() # >> 100
```

* In fact, initializer op is the assign op that assigns the initial value to the variable.

For example, 
```
# create a variable whose original value s 2
my_var = tf.Variable(2, name="my_var")

# assign a * 2 to a and call that op a a_times_two
my_var_times_two = my_var.assign(2 * my_var)

with tf.Session() as sess:
    sess.run(my_var.initializer) # no return value
    print my_var.eval() # >> 2
    print sess.run(my_var_times_two) # >> 4
    print sess.run(my_var_times_two) # >> 8
    print sess.run(my_var_times_two) # >> 16
```

**5. Assign_add and Assign_sub**

Increment or decrement the value.

But note that `assign_add` and `assign_sub` ops can't initialize the variables for you, so you have to call the initializer before you call these two ops.

**6. Each session maintains its own copy of variable**

> Each session maintains its own copy of variable.

For exmaple,

```
W = tf.Variable(10)

sess1 = tf.Session()
sess2 = tf.Session()

sess1.run(W.initializer)
sess2.run(W.initializer)

print sess1.run(W.assign_add(10)) # >> 20
print sess2.run(W.assign_sub(2)) # >> 8

sess1.close()
sess2.close()
```

## 8 Placeholders
---

**1. tf.placeholder**

**`tf.placeholder(dtype, shape=None, name=None)`**

`feed_dict`: document of all placeholders. Put the tensor as the key.

* `shape=None` means that tensor of any shape will be accepted as value for placeholder.
* `shape=None` is easy to construct graphs, but nightmares for debugging.
* `shape=None` also breaks all following shape inference, which makes many ops not work because they expect certain rank.

**2. Feed multiple data points in**

```
with tf.Session() as sess:
   for a_value in list_of_values_for_a:
       print sess.run(c, {a: a_value})
```

## 9 Optimizer
---

```
tf.train.GradientDescentOptimizer
tf.train.AdagradientOptimizer
tf.train.MomentumOptimizer
tf.train.AdamOptimizer
tf.train.ProximalGradientDescentOptimizer
tf.train.ProximalAdagradientOptimizer
tf.train.RMSPropOptimizer
And more
```

## 10 Define Loss Functions As a Function
---

Use tf.less and tf.select to implement complex functions.

```
# Implement Huber Loss
def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.select (condition, small_res, large_res)
```

### 11 Test The Model
---

```
n_batches = int (test.num_examples/batch_size)
total_correct_preds = 0
for i in xrange(n_batches):
    X_batch, Y_batch = test.next_batch(batch_size)
    loss_batch, logits_batch = sess.run([loss, logits], feed_dict={X:X_batch, Y:Y_batch})
    preds = tf.nn.softmax(logits_batch)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
    num_of_correct_preds = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
    total_correct_preds += sess.run(num_of_correct_preds)
    
print 'Accuracy {0}'.format(total_correct_preds/test.num_examples)
```

## 12 Name Scope
---

```
with tf.name_scope(name):
```
