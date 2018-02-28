import torch
from torch.autograd import Variable

from allennlp.nn.util import logsumexp

# span_logits is (batch_size, num_paragraphs, num_tokens)
# paragraph_mask is (batch_size, num_paragraphs)
# actual_spans is (batch_size, num_paragraphs, num_spans)
# actual_spans_mask is (batch_size, num_paragraphs, num_spans)
class SharedNormLoss(torch.nn.Module):
    def forward(self,
                span_logits: Variable,
                paragraph_mask: Variable,
                actual_spans: Variable,
                actual_spans_mask: Variable) -> Variable:
        batch_size, num_paragraphs, num_tokens = span_logits.size()
        _, _, num_spans = actual_spans.size()

        # print(f"batch size: { batch_size}, num_paragraphs: {num_paragraphs}, num_tokens: {num_tokens}, num_spans: {num_spans}")

        # (batch_size, num_paragraphs, num_tokens), with 0s for missing paragraphs
        masked_span_logits = span_logits * paragraph_mask.unsqueeze(-1)

        # (batch_size, num_paragraphs * num_tokens)
        masked_span_logits = masked_span_logits.view(batch_size, num_paragraphs * num_tokens)

        # logsumexp across paragraphs * tokens then sum across batch
        denominator = logsumexp(masked_span_logits).sum()

        # actual_spans (batch_size, num_paragraphs, num_spans)
        # span_logits  (batch_size, num_paragraphs, num_tokens)
        #
        # want a (batch_size, num_paragraphs, num_spans) tensor of span_logits
        # that means we need to gather from span_logits

        # replace -1s with 0s (TODO: this feels hacky)
        masked_actual_spans = torch.max(actual_spans, 0 * actual_spans)
        actual_span_logits = torch.gather(span_logits, 2, masked_actual_spans)

        # then apply mask, still (batch_size, num_paragraphs, num_spans)
        actual_span_logits.masked_fill_((1 - actual_spans_mask).byte(), -10000)

        # (batch_size, num_paragraphs * num_spans)
        actual_span_logits = actual_span_logits.view(batch_size, num_paragraphs * num_spans)
        numerator = logsumexp(actual_span_logits).sum()

        return denominator - numerator
