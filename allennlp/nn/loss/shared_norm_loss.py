import torch
from torch.autograd import Variable

from allennlp.nn.util import logsumexp

# span_logits is (batch_size, num_paragraphs, num_tokens)
# paragraph_mask is (batch_size, num_paragraphs)
# spans is (batch_size, num_paragraphs, num_spans)
# spans_mask is (batch_size, num_paragraphs, num_spans)
class SharedNormLoss(torch.nn.Module):
    def forward(self,
                span_logits: Variable,
                paragraph_mask: Variable,
                spans: Variable,
                spans_mask: Variable) -> Variable:


        return 0


    # # scores : (batch_size, num_paragraphs, num_spans)
    # # span_idxs: (batch_size, num_paragraphs, num_ans)
    # def forward(self,
    #             scores: Variable,
    #             paragraph_mask: Variable,
    #             span_idxs: Variable,
    #             answer_mask: Variable) -> Variable:
    #     # pylint: disable=arguments-differ,unused-argument
    #     # TODO(joelgrus): why is paragraph_mask unused?
    #     masked_span_idxs = (span_idxs * answer_mask.long()).squeeze(-1)
    #     pos_scores = torch.gather(scores, 2, masked_span_idxs) # (batch_size, num_paras, num_ans)

    #     pos_scores.data.masked_fill_(1 - answer_mask.data.byte(), -float('inf'))
    #     pos_log_sum_exp = logsumexp(pos_scores)
    #     all_log_sum_exp = logsumexp(scores)
    #     loss = -torch.mean(pos_log_sum_exp - all_log_sum_exp)
    #     return loss
