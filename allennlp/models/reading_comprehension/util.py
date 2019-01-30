import torch


def get_best_span(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor) -> torch.Tensor:
    if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
        raise ValueError("Input shapes must be (batch_size, passage_length)")
    batch_size, passage_length = span_start_logits.size()
    max_span_log_prob = [-1e20] * batch_size
    span_start_argmax = [0] * batch_size
    best_word_span = span_start_logits.new_zeros((batch_size, 2), dtype=torch.long)

    span_start_logits = span_start_logits.detach().cpu().numpy()
    span_end_logits = span_end_logits.detach().cpu().numpy()

    for b in range(batch_size):  # pylint: disable=invalid-name
        for j in range(passage_length):
            val1 = span_start_logits[b, span_start_argmax[b]]
            if val1 < span_start_logits[b, j]:
                span_start_argmax[b] = j
                val1 = span_start_logits[b, j]

            val2 = span_end_logits[b, j]

            if val1 + val2 > max_span_log_prob[b]:
                best_word_span[b, 0] = span_start_argmax[b]
                best_word_span[b, 1] = j
                max_span_log_prob[b] = val1 + val2
    return best_word_span
