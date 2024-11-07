---
layout: post
title: How to Write Good Python Code
date: 2023-02-07 15:09:00
description: Let's look at the basic rules for writing great Python code
tags: python, coding
categories: old-posts
featured: true
toc:
  sidebar: left
---

Python is a fantastic programming language!

It can be used for many things, like building websites, exploring data, and teaching machines to learn.
If you already know Python or are just beginning, writing code that is strong, easy to read, and easy to keep up with is important.
In this article, we’ll look at the basic rules for writing great Python code and share some tips to help you make your programs even better.

```yml
toc:
  sidebar: left
```

## 📚 Use Meaningful Naming Conventions
One of the most important aspects of good Python code is meaningful naming conventions.
Choosing descriptive and concise names for variables, functions, and classes can help make your code more readable and understandable.
Using proper naming conventions can also help you avoid naming conflicts, reduce the risk of errors, and simplify maintenance.

For example, these are bad variable names:
```python
x = 5
y = 10
z = x + y
w = z * 2
```

And these are better ones:
```python
first_number = 5
second_number = 10
sum_of_numbers = first_number + second_number
double_sum = sum_of_numbers * 2
```

This includes using proper indentation, white space, line breaks, and following a code style guide like the PEP 8 style guide.

Clear, organized code makes it easier to understand and modify and reduces the risk of errors.

Here’s an example of lousy code organization:

```python
def calc_sum(a, b, c, d):
    return a + b + c + d
def calc_diff(a, b, c, d):
    return a - b - c - d
def calc_product(a, b, c, d):
    return a * b * c * d
def calc_quotient(a, b, c, d):
    return a / b / c / d
def process_data(a, b, c, d):
    s = calc_sum(a, b, c, d)
    d = calc_diff(a, b, c, d)
    p = calc_product(a, b, c, d)
    q = calc_quotient(a, b, c, d)
    print("Sum: ", s)
    print("Difference: ", d)
    print("Product: ", p)
    print("Quotient: ", q)
```

And here’s an example of good code organization:

```python
def calc_sum(a, b, c, d):
    return a + b + c + d

def calc_diff(a, b, c, d):
    return a - b - c - d

def calc_product(a, b, c, d):
    return a * b * c * d

def calc_quotient(a, b, c, d):
    return a / b / c / d

def process_data(a, b, c, d):
    s = calc_sum(a, b, c, d)
    d = calc_diff(a, b, c, d)
    p = calc_product(a, b, c, d)
    q = calc_quotient(a, b, c, d)
    
    results = {
        "Sum": s,
        "Difference": d,
        "Product": p,
        "Quotient": q
    }
    
    return results
```

## 💬 Write Comments
Adding comments to your code is a great way to explain what it does and provide context for other developers.
Comments should be used to explain complex code, provide additional information about the purpose of the code, and describe your thought process.
Writing comments can also help you better understand your code when you return to it later.

```python
def calc_sum(a, b):
    # function to calculate sum
    c = a + b
    return c

# this function calculates sum of two numbers
def calc_difference(a, b):
    return a - b
```

The comments are not very descriptive or helpful in understanding the purpose of the functions.
The first comment is trivial and adds no additional information. The second comment repeats what the function name already tells us.

```python
def calc_sum(a, b):
    """
    This function calculates the sum of two numbers `a` and `b`.
    The function takes in two keyword arguments, `a` and `b`, and returns their sum.
    """
    # calculate the sum of `a` and `b`
    c = a + b
    return c

def calc_difference(a, b):
    """
    This function calculates the difference between two numbers `a` and `b`.
    The function takes in two keyword arguments, `a` and `b`, and returns the difference of `a` and `b`.
    """
    # calculate the difference between `a` and `b`
    return a - b
```

Here, the comments provide a clear and concise explanation of the purpose and behaviour of each function.
The use of docstrings makes it easy to understand what the functions do and what arguments they take in. This makes the code more readable and maintainable.