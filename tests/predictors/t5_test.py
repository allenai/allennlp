from allennlp.predictors.t5 import T5Predictor


def test_t5_predictor():
    p = T5Predictor.from_pretrained("t5-small")
    o = p.predict("translate English to German: That is good")
    assert o["predicted_text"][0] == "Das ist gut."
