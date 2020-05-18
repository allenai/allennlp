This is a docstring.

<a name=".scripts.tests.py2md.basic_example.SOME_GLOBAL_VAR"></a>
## SOME\_GLOBAL\_VAR

```python
SOME_GLOBAL_VAR = "Ahhhh I'm a global var!!"
```

This is a global var.

<a name=".scripts.tests.py2md.basic_example.func_with_no_args"></a>
## func\_with\_no\_args

```python
def func_with_no_args()
```

This function has no args.

<a name=".scripts.tests.py2md.basic_example.func_with_args"></a>
## func\_with\_args

```python
def func_with_args(a: int, b: int, c: int = 3) -> int
```

This function has some args.

<strong>Parameters</strong>


- __a__ : `int` <br>
    A number.
- __b__ : `int` <br>
    Another number.
- __c__ : `int`, optional (default = `3`) <br>
    Yet another number.

<strong>Returns</strong>


- `int` <br>
    The result of `a + b * c`.

<a name=".scripts.tests.py2md.basic_example.SomeClass"></a>
## SomeClass Objects

```python
class SomeClass():
 | def __init__(self) -> None
```

I'm a class!

<strong>Paramaters</strong>


x : `float`
    This attribute is called `x`.

<a name=".scripts.tests.py2md.basic_example.SomeClass.some_class_level_variable"></a>
### some\_class\_level\_variable

```python
some_class_level_variable = 1
```

This is how you document a class-level variable.

<a name=".scripts.tests.py2md.basic_example.SomeClass.some_class_level_var_with_type"></a>
### some\_class\_level\_var\_with\_type

```python
some_class_level_var_with_type = 1
```

<a name=".scripts.tests.py2md.basic_example.SomeClass.some_method"></a>
### some\_method

```python
 | def some_method(self) -> None
```

I'm a method!

But I don't do anything.

<strong>Returns</strong>


- `None` <br>

<a name=".scripts.tests.py2md.basic_example.SomeClass.method_with_alternative_return_section"></a>
### method\_with\_alternative\_return\_section

```python
 | def method_with_alternative_return_section(self) -> int
```

Another method.

<strong>Returns</strong>


- A completely arbitrary number. <br>

<a name=".scripts.tests.py2md.basic_example.SomeClass.method_with_alternative_return_section3"></a>
### method\_with\_alternative\_return\_section3

```python
 | def method_with_alternative_return_section3(self) -> int
```

Another method.

<strong>Returns</strong>


- __number__ : `int` <br>
    A completely arbitrary number.

<a name=".scripts.tests.py2md.basic_example.AnotherClassWithReallyLongConstructor"></a>
## AnotherClassWithReallyLongConstructor Objects

```python
class AnotherClassWithReallyLongConstructor():
 | def __init__(
 |     self,
 |     a_really_long_argument_name: int = 0,
 |     another_long_name: float = 2,
 |     these_variable_names_are_terrible: str = "yea I know",
 |     **kwargs
 | ) -> None
```

