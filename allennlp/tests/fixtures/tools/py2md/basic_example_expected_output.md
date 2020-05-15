[ *allennlp**.tests**.fixtures**.tools**.py2md***.basic_example** ]

---

This is a docstring.

<a name=".allennlp.tests.fixtures.tools.py2md.basic_example.SOME_GLOBAL_VAR"></a>
## SOME\_GLOBAL\_VAR

```python
SOME_GLOBAL_VAR = "Ahhhh I'm a global var!!"
```

This is a global var.

<a name=".allennlp.tests.fixtures.tools.py2md.basic_example.func_with_no_args"></a>
## func\_with\_no\_args

```python
def func_with_no_args()
```

This function has no args.

<a name=".allennlp.tests.fixtures.tools.py2md.basic_example.func_with_args"></a>
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

<a name=".allennlp.tests.fixtures.tools.py2md.basic_example.SomeClass"></a>
## SomeClass

```python
class SomeClass():
 | def __init__(self) -> None
```

I'm a class!

<strong>Attributes</strong>


- __x__ : `float` <br>
    This attribute is called `x`.

<a name=".allennlp.tests.fixtures.tools.py2md.basic_example.SomeClass.some_method"></a>
### some\_method

```python
 | def some_method(self) -> None
```

I'm a method!

But I don't do anything.

<strong>Returns</strong>


- `None` <br>

<a name=".allennlp.tests.fixtures.tools.py2md.basic_example.SomeClass.method_with_alternative_return_section"></a>
### method\_with\_alternative\_return\_section

```python
 | def method_with_alternative_return_section(self) -> int
```

Another method.

<strong>Returns</strong>


- A completely arbitrary number. <br>

<a name=".allennlp.tests.fixtures.tools.py2md.basic_example.SomeClass.method_with_alternative_return_section2"></a>
### method\_with\_alternative\_return\_section2

```python
 | def method_with_alternative_return_section2(self) -> int
```

Another method.

<strong>Returns</strong>


- __number__ : `int` <br>
    A completely arbitrary number.

<a name=".allennlp.tests.fixtures.tools.py2md.basic_example.AnotherClassWithReallyLongConstructor"></a>
## AnotherClassWithReallyLongConstructor

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

