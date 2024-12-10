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

In this bogpost, we’ll look at the basic rules for writing great Python code and share some tips to help you make your programs even better.

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

The comments provide a clear and concise explanation of the purpose and behaviour of each function.

The use of docstrings makes it easy to understand what the functions do and what arguments they take in. This makes the code more readable and maintainable.

## 🧰 Use Modules and Packages

Modules and packages are a great way to organize your code into reusable blocks. 

They allow you to group related code together and make it easier to manage, understand, and maintain.

The Python Standard Library is an good resource for finding pre-existing modules and packages. You can import it into your programs to save time and effort.

Consider a project to build a simple weather application that provides a given city’s current temperature and conditions. We can structure the project as follows:

```
weather_app/
    __init__.py
    weather.py
    utils/
        __init__.py
        api.py
        data_processing.py
```

`weather.py` is the main module that the user interacts with, which provides a single function to get the current weather information.

```python
def get_current_weather(city: str) -> dict:
    """
    Gets the current weather information for the given city.

    Args:
        city (str): The city for which to get the weather information.

    Returns:
        dict: The weather information for the given city.
    """
    weather_data = utils.api.get_weather_data(city)
    processed_data = utils.data_processing.process_weather_data(weather_data)

    return processed_data
```

The utils package contains two modules, `api.py` and `data_processing.py`, which contain helper functions to retrieve the raw weather data from an API and to process the raw data into a more readable format, respectively. 

These modules can be reused across different projects, so it makes sense to organize them into a separate package.

```python
# api.py
def get_weather_data(city: str) -> dict:
    """
    Retrieves the raw weather data for the given city.

    Args:
        city (str): The city for which to retrieve the weather data.

    Returns:
        dict: The raw weather data for the given city.
    """
    # code to retrieve data from API
    return raw_data

# data_processing.py
def process_weather_data(raw_data: dict) -> dict:
    """
    Processes the raw weather data into a more readable format.

    Args:
        raw_data (dict): The raw weather data.

    Returns:
        dict: The processed weather data.
    """
    # code to process data
    return processed_data
```

## 🧪 Test Your Code
Testing your code helps you catch bugs and ensure that your code works as expected.

Writing test cases is also an good way to document your code and help others understand it. Try all possible scenarios when testing your code, including edge cases and error conditions.

Consider a module `calculator.py` that implements a simple calculator with basic arithmetic operations. We can write test cases for each operation using a testing framework such as `unittest`.

```python
import unittest
import calculator

class TestCalculator(unittest.TestCase):
    def test_addition(self):
        result = calculator.add(2, 3)
        self.assertEqual(result, 5)

    def test_subtraction(self):
        result = calculator.subtract(5, 3)
        self.assertEqual(result, 2)

    def test_multiplication(self):
        result = calculator.multiply(2, 3)
        self.assertEqual(result, 6)

    def test_division(self):
        result = calculator.divide(6, 2)
        self.assertEqual(result, 3)

if __name__ == '__main__':
    unittest.main()
```

Each test case tests a single operation in the calculator module and uses the `assertEqual` method to verify that the result of the operation is as expected. 

If any test fails, an error will be raised, and the test result will be reported as failed.

For debugging we can use the `print` statement to print the intermediate results or the values of variables in the code, or use a debugger such as `pdb` to step through the code and inspect the values of variables.

```python
import calculator
import pdb

result = calculator.add(2, 3)
pdb.set_trace() # Set a breakpoint
print(result)
```

## 📜 Document Your Code

Documenting your code with docstrings can help others understand what it does and how it works. 

Docstrings should provide a high-level overview of the code, including its purpose, usage, and limitations. 

They should also be written in a clear and natural language style.

```python
class Circle:
    """
    Class to represent a circle with a given radius.

    Attributes:
        radius (float): The radius of the circle.
    """

    def __init__(self, radius: float):
        """
        Initializes the Circle class with a given radius.

        Args:
            radius (float): The radius of the circle.
        """
        self.radius = radius

    def area(self) -> float:
        """
        Calculates the area of the circle.

        Returns:
            float: The area of the circle.
        """
        return 3.14 * (self.radius ** 2)

    def circumference(self) -> float:
        """
        Calculates the circumference of the circle.

        Returns:
            float: The circumference of the circle.
        """
        return 2 * 3.14 * self.radius
```

The class has a docstring explaining its purpose and the attributes it has. 

Each method has its docstring explaining what it does and what arguments it takes and returns. 

This makes the code easier to understand and maintain and more accessible for others to use and build upon.

## 💥 Handle Exceptions Gracefully

Handling exceptions in your code is essential for ensuring that it continues to run even when unexpected events occur. 

Use `try` and `except` statements to handle exceptions and provide helpful error messages that explain what went wrong and how to fix it.

```python
try:
    # code that may raise an exception
    result = 10 / 0
except ZeroDivisionError as error:
    # handle the exception
    print("An error occurred:", error)
    print("Please provide a non-zero value for division")
```

The code inside the `try` block may raise a `ZeroDivisionError` exception. 

The `except` block handles the exception and prints a helpful error message to the user. 

This way, the program can continue running even when an unexpected error occurs.

```python
try:
    # code that may raise an exception
    with open("file.txt") as file:
        data = file.read()
except FileNotFoundError as error:
    # handle the exception
    print("An error occurred:", error)
    print("Please provide a valid file path")
except Exception as error:
    # handle any other exceptions
    print("An unexpected error occurred:", error)
```

In this example, the code inside the try block may raise a `FileNotFoundError` or any other exception.

The first `except` block handles the FileNotFoundError and provides a helpful error message for the user. 

The second `except` block handles any other exceptions that may occur and provides a generic error message. 

This way the program can continue running even when unexpected errors occur and provide helpful error messages to the user.

## 🔑 Use Keyword Arguments

Keyword arguments are a powerful feature of Python that allows you to specify default values for function arguments and make your code more readable and flexible.

Using keyword arguments can also help you reduce the number of lines of code in your programs and make them easier to understand.

```python
def greet(name, message="Hello"):
    print(f"{message}, {name}!")

greet("John") # Output: Hello, John!
greet("Jane", message="Hi") # Output: Hi, Jane!
```

In this example, the greet function takes in two arguments: name and message. The message argument has a default value of "Hello".

When we call `greet("John")`, the default value of `"Hello"` is used for the message argument. But when we call `greet("Jane", message="Hi")`, the keyword argument is used instead, and the output is `"Hi, Jane!"`.

## 🧘‍♀️ Follow the Zen of Python

The Zen of Python is a collection of principles and guidelines for writing good Python code. 

It includes tips on writing simple, clear, and maintainable code and advice on choosing between different solutions.

```python
import this

def sort_data(data):
    # Simple is better than complex
    data.sort()
    return data

# Readability counts
def add(a, b):
    # Explicit is better than implicit
    return a + b

# Flat is better than nested
def flatten(lists):
    result = []
    for sublist in lists:
        for item in sublist:
            result.append(item)
    return result

# Use meaningful names
def calculate_average_score(scores):
    total = 0
    count = 0
    for score in scores:
        total += score
        count += 1
    # One obvious way to do it
    return total / count
```

We follow the Zen of Python by:

- Writing straightforward code (e.g. the `sort_data` function)
- Choosing meaningful names for variables and functions (e.g. `calculate_average_score`)
- Keeping the code flat and avoiding nested structures where possible (e.g. the flatten function)
- Being explicit and transparent in our code (e.g. using return statements)

## 🛠 Refactor Your Code Regularly

Refactoring is improving the structure and quality of your code without changing its external behaviour. 

It can help you identify areas that need improvement and make your code more maintainable over time. This can be especially important in projects with a long lifespan or requiring continuous updates.

So you can simplify complex sections, make your code more efficient, and eliminate any redundant or unnecessary parts. You can also take advantage of new features or libraries that have become available since you wrote the original code.

```python
# Original code
def calculate_sum(numbers):
    result = 0
    for number in numbers:
        result += number
    return result

# Refactored code
def calculate_sum(numbers):
    return sum(numbers)
```

We have refactored the `calculate_sum` function to use the built-in sum function instead of manually iterating over the numbers and adding them up. This code is more efficient and readable and takes advantage of a built-in feature of Python that can perform the same calculation.