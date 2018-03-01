# from typing import List, Tuple
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import pairwise_distances

# import numpy as np
# import random

# from allennlp.data.tokenizers.token import Token

# class Paragraph:
#     def __init__(self, text: str = '', tokens: List[Token] = None) -> None:
#         self.text = text
#         self.tokens = [] if tokens is None else tokens
#         self.answers: List[str] = None

#     def __len__(self) -> int:
#         return len(self.tokens)

#     def _assert_consistent(self) -> None:
#         for token in self.tokens:
#             start = token.idx
#             end = token.idx + len(token.text)
#             if token.text != self.text[start:end]:
#                 assert False

#     def add(self, other: 'Paragraph') -> None:
#         assert isinstance(other, Paragraph)
#         for token in other.tokens:
#             token.idx += len(self.text) + 1
#         self.text += ' ' + other.text
#         self.tokens += other.tokens
#         self._assert_consistent()

#     def truncate(self, max_tokens: int) -> None:
#         self.tokens = self.tokens[:max_tokens]

#         last_token = self.tokens[-1]
#         new_end_pos = last_token.idx + len(last_token.text)
#         self.text = self.text[:new_end_pos]
#         self._assert_consistent()

#     @property
#     def offsets(self) -> List[Tuple[int, int]]:
#         return [(token.idx, token.idx + len(token.text)) for token in self.tokens]


# class TopTfIdf:
#     def __init__(self, stop_words: str) -> None:
#         self.tfidf = TfidfVectorizer(strip_accents="unicode", stop_words=stop_words)

#     def score(self, question: str, paragraphs: List[Paragraph]):
#         paragraph_texts = [para.text for para in paragraphs]
#         try:
#             # (num_paragraphs, num_features)
#             para_features = self.tfidf.fit_transform(paragraph_texts)
#             # (1, num_features)
#             q_features = self.tfidf.transform([question])
#         except ValueError:
#             # (num_paragraphs,)
#             return np.array([0.0] * len(paragraph_texts))
#         # pairwise_distances is (1, num_paragraphs), after ravel is (num_paragraphs,)
#         dists = pairwise_distances(q_features, para_features, "cosine").ravel()
#         return dists

# def sort(paragraphs: List[Paragraph], scores: np.ndarray) -> List[Paragraph]:
#     # In case of ties, use the earlier paragraph. Here np.lexsort
#     # will return the indices for sorting by `scores` then by paragraph number
#     sorted_ix = np.lexsort((range(len(paragraphs)), scores))
#     return [paragraphs[i] for i in sorted_ix]

# class ParagraphSampler:
#     def get_permutations(self, instances: List[Paragraph]) -> List[Paragraph]:
#         raise NotImplementedError()

#     def sample(self, instances: List[Paragraph]) -> List[Paragraph]:
#         raise NotImplementedError()

# # class TopKParagraphSampler(ParagraphSampler):
# #     def __init__(self, k: int, only_with_answers: bool, num_per_sample: int = None):
# #         self.k = k
# #         self.only_with_answers = only_with_answers
# #         self.num_per_sample = k if num_per_sample is None else num_per_sample

# #     def get_permutations(self, instances: List[Paragraph]):
# #         if self.num_per_sample == self.k:
# #             return [ordered_paragraphs]
# #         else:
# #             raise NotImplementedError()

# #     def selected_samples(self, paragraphs, scores):
# #         assert len(paragraphs) == len(scores)
# #         ordered_paragraphs = sort(paragraphs, scores)
# #         # sorted_with_scores = reversed(sorted(zip(paragraphs, scores), key=lambda x: x[1]))
# #         # ordered_paragraphs = [paragraph for (paragraph, score) in sorted_with_scores]
# #         if self.only_with_answers:
# #             # remove without answers
# #             ordered_paragraphs = [paragraph for paragraph in paragraphs if paragraph.has_answers()]
# #         ordered_paragraphs = ordered_paragraphs[:self.k]
# #         return self.get_permutations(ordered_paragraphs)

# #     def sample(self, instances):
# #         raise NotImplementedError()

# # class FirstNTokensParagraphSampler():
# #     def __init__(self, num_tokens):
# #         super(FirstNTokensParagraphSampler).__init__()
# #         self.num_tokens = num_tokens

# #     def selected_samples(self, paragraphs):
# #         first_paragraph = paragraphs[0]
# #         return first_paragraph

# #     def sample(self, instances):
# #         raise NotImplementedError()

# # class StratifyParagraphSampler(ParagraphSampler):
# #     def __init__(self, oversample_first_answer_paragraph=True, sample_upper_limit=4):
# #         super(StratifyParagraphSampler).__init__()
# #         self.oversample_first_answer_paragraph = oversample_first_answer_paragraph
# #         self.sample_upper_limit = sample_upper_limit

# #     # Returns list of samples. Each sample contains a list of sample_upper_limit paragraphs.
# #     def selected_samples(self, paragraphs, scores):
# #         assert len(paragraphs) == len(scores)
# #         ordered_paragraphs = sort(paragraphs, scores)
# #         # sorted_with_scores = reversed(sorted(zip(paragraphs, scores), key=lambda x: x[1]))
# #         # ordered_paragraphs = [paragraph for (paragraph, score) in sorted_with_scores]
# #         ordered_paragraphs = ordered_paragraphs[:self.sample_upper_limit]
# #         # paragraphs_with_answers = [paragraph for paragraph in ordered_paragraphs if paragraph.has_answers()]
# #         paragraphs_with_answers = [i for i in range(len(ordered_paragraphs)) if ordered_paragraphs[i].has_answers()]
# #         # oversample first answer paragraph if we have more than 1 answer paragraphs
# #         if len(paragraphs_with_answers) == 0:
# #             return []
# #         elif len(paragraphs_with_answers) == 1 and len(ordered_paragraphs) == 1:
# #             return [[ordered_paragraphs[paragraphs_with_answers[0]]]]
# #         if self.oversample_first_answer_paragraph and len(paragraphs_with_answers) > 1:
# #             for paragraph_i in range(len(ordered_paragraphs)):
# #                 if ordered_paragraphs[paragraph_i].has_answers():
# #                     paragraphs_with_answers += [paragraph_i]
# #                     break
# #         # permutations
# #         permutations = []
# #         # bug: removes the effect of oversampling

# #         if len(paragraphs_with_answers) > 0:
# #             # single_para_has_answers = len(paragraphs_with_answers) == 1 and len(paragraphs_with_answers) == len(ordered_paragraphs)
# #             for answer_paragraph_i in paragraphs_with_answers:
# #                 for any_paragraph_i in range(len(ordered_paragraphs)):
# #                     # unordered_pair = frozenset({answer_paragraph_i, any_paragraph_i})
# #                     if answer_paragraph_i != any_paragraph_i:
# #                         permutations.append([ordered_paragraphs[answer_paragraph_i], ordered_paragraphs[any_paragraph_i]])
# #                         # permutations_i.add(unordered_pair)
# #         return permutations


# #     def sample(self, instances):
# #         raise NotImplementedError()
