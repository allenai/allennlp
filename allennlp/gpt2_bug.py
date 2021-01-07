from allennlp.interpret.saliency_interpreters import SimpleGradient
from allennlp_models.pretrained import load_predictor


predictor = load_predictor(
    "lm-next-token-lm-gpt2",
    overrides={
        "dataset_reader.max_tokens": 512,
        "model.beam_search_generator": {
            "type": "transformer",
            "beam_search": {
                "end_index": 50256,
                "max_steps": 5,
                "beam_size": 5,
            },
        },
    },
)
# bp()
interpreter = SimpleGradient(predictor)
interpreter.saliency_interpret_from_json({"sentence": "Hi there"})
