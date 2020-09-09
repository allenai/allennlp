
import torch

from allennlp.common import FromParams

class BiModalEncoder(torch.nn.Module, FromParams):
	def __init__(self)