"""
A `SimilarityFunction` takes a pair of tensors with the same shape, and computes a similarity
function on the vectors in the last dimension.
"""
from allennlp.modules.similarity_functions.bilinear import BilinearSimilarity
from allennlp.modules.similarity_functions.cosine import CosineSimilarity
from allennlp.modules.similarity_functions.dot_product import DotProductSimilarity
from allennlp.modules.similarity_functions.linear import LinearSimilarity
from allennlp.modules.similarity_functions.multiheaded import MultiHeadedSimilarity
from allennlp.modules.similarity_functions.similarity_function import SimilarityFunction
