# AllenNLP Style Guide

Our highest priority for code style is that our code be easily readable to someone new to the
codebase.  Deep learning is easy to get wrong, and we want our code to be easy enough to read that
someone looking at it can be thinking about our modeling decisions, not trying to understand what
is going on.

To that end, we use descriptive names, we use type annotations, and we write coherent docstrings.
In code that manipulates tensors, most lines that compute a tensor have a comment describing the
tensor's shape.  When there's an interesting or important modeling decision in the code, we write
a comment about it (either in-line or in an appropriate docstring).

## Docstrings

All reasonably complex public methods should have docstrings describing their basic function, their
inputs and their outputs.  Private methods should also most often have docstrings, so that people
who read your code know what the method is supposed to do.  The basic outline we use for docstrings
is: (1) a brief description of what the method does, sometimes also including how or why the method
does it, (2) the parameters / arguments to the method, (3) the return value of the method, if any.
If the method is particularly simple or the arguments are obvious, (2) and (3) can be omitted.  We
use [numpydoc](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt) to produce
our documentation, so function arguments and return values should be formatted as shown in the link
(or as seen in just about any model or module in the codebase).  We treat the class docstring as
the documentation for `__init__` methods, giving parameters there and omitting any docstring on the
constructor itself.  For model / module constructors and methods like `forward`, _always_ include
the parameters and return values (when there is one) in the docstring.

## Code format

We use pylint to enforce some basic consistency in formatting.  Those formatting guidelines roughly
follow [Google's python style
guide](https://google.github.io/styleguide/pyguide.html#Python_Style_Rules), with a few notable
exceptions.  In particular, because we use type annotations and descriptive variable names, we use
100-character lines instead of 80-character lines, and it's ok to go over sometimes in code.
Pylint enforces a hard boundary of 115 characters, but you should try to stay under 100 characters
most of the time (in particular, comments and docstrings should wrap to the next line at no more
than 100 characters).  Additionally, we use `numpydoc` and `sphinx` for building our docs, so
Google's docstring formats don't apply.

## Naming

We follow Google's [general naming
rules](https://google.github.io/styleguide/cppguide.html#General_Naming_Rules), and their
[definition of camel case](https://google.github.io/styleguide/javaguide.html#s5.3-camel-case).

## Module layout and imports

To keep files from getting too big, we typically have one class per file, though small classes
that are inseparable from a companion class can also go in the same file (often these will be
private classes).

To avoid verbosity when importing classes structured this way, classes should be imported from
their module's `__init__.py`.  For example, the `Dataset` class is in `allennlp/data/dataset.py`,
but `allennlp/data/__init__.py` imports the class, so that you can just do `from allennlp.data
import Dataset`.

Abstract classes typically go in a module containing the abstract class and all built-in
implementations.  This includes things like `Field` (in `allennlp.data.fields`), `Seq2SeqEncoder`
(in `allennlp.modules.seq2seq_encoders`), and many others.  In these cases, the abstract class
should be imported into the module _above_, so that you can do, e.g., `from allennlp.data import
Field`.  Concrete implementations follow the same layout as above: `from allennlp.data.fields
import TextField`.

Imports should be formatted at the top of the file, following [PEP 8's
recommendations](https://www.python.org/dev/peps/pep-0008/#imports): three sections (standard
library, third-party libraries, internal imports), each sorted and separated by a blank line.

## Conclusion

Some of the conventions we've adopted are arbitrary (e.g., other definitions of camel case are
also valid), but we stick to them to keep a consistent style throughout the codebase, which makes
it easier to read and maintain.
