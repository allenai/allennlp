# allennlp.common.testing.test_case

## AllenNlpTestCase
```python
AllenNlpTestCase(self, methodName='runTest')
```

A custom subclass of :class:`~unittest.TestCase` that disables some of the
more verbose AllenNLP logging and that creates and destroys a temp directory
as a test fixture.

