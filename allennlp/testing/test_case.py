# pylint: disable=invalid-name,protected-access
from copy import deepcopy
from unittest import TestCase
import codecs
import gzip
import logging
import os
import shutil

import numpy
from numpy.testing import assert_allclose

from ..common.checks import log_pytorch_version_info
from ..common.params import Params


class AllenNlpTestCase(TestCase):  # pylint: disable=too-many-public-methods
    TEST_DIR = './TMP_TEST/'
    TRAIN_FILE = TEST_DIR + 'train_file'
    VALIDATION_FILE = TEST_DIR + 'validation_file'
    TEST_FILE = TEST_DIR + 'test_file'
    TRAIN_BACKGROUND = TEST_DIR + 'train_background'
    VALIDATION_BACKGROUND = TEST_DIR + 'validation_background'
    SNLI_FILE = TEST_DIR + 'snli_file'
    PRETRAINED_VECTORS_FILE = TEST_DIR + 'pretrained_glove_vectors_file'
    PRETRAINED_VECTORS_GZIP = TEST_DIR + 'pretrained_glove_vectors_file.gz'

    def setUp(self):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            level=logging.DEBUG)
        log_pytorch_version_info()
        os.makedirs(self.TEST_DIR, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.TEST_DIR)

    def get_trainer_params(self, additional_arguments=None):
        params = Params({})
        params['save_models'] = False
        params['model_serialization_prefix'] = self.TEST_DIR
        params['train_files'] = [self.TRAIN_FILE]
        params['validation_files'] = [self.VALIDATION_FILE]
        params['num_epochs'] = 1

        if additional_arguments:
            for key, value in additional_arguments.items():
                params[key] = deepcopy(value)
        return params

    def get_model_params(self, additional_arguments=None):
        params = Params({})
        params['save_models'] = False
        params['model_serialization_prefix'] = self.TEST_DIR
        params['train_files'] = [self.TRAIN_FILE]
        params['validation_files'] = [self.VALIDATION_FILE]
        params['embeddings'] = {'words': {'dimension': 6}, 'characters': {'dimension': 2}}
        params['encoder'] = {"default": {'type': 'bow'}}
        params['num_epochs'] = 1
        params['validation_split'] = 0.0

        if additional_arguments:
            for key, value in additional_arguments.items():
                params[key] = deepcopy(value)
        return params

    def get_model(self, model_class, additional_arguments=None):
        params = self.get_model_params(additional_arguments)
        return model_class(params)

    def ensure_model_trains_and_loads(self, model_class, args: Params):
        args['save_models'] = True
        # Our loading tests work better if you're not using data generators.  Unless you
        # specifically request it in your test, we'll avoid using them here, and if you _do_ use
        # them, we'll skip some of the stuff below that isn't compatible.
        args.setdefault('data_generator', None)
        model = self.get_model(model_class, args)
        model.train()

        # load the model that we serialized
        loaded_model = self.get_model(model_class, args)
        loaded_model.load_model()

        # verify that original model and the loaded model predict the same outputs
        if model._uses_data_generators():
            # We shuffle the data in the data generator.  Instead of making that logic more
            # complicated, we'll just pass on the loading tests here.  See comment above.
            pass
        else:
            model_predictions = model.model.predict(model.validation_arrays[0])
            loaded_model_predictions = loaded_model.model.predict(model.validation_arrays[0])

            for model_prediction, loaded_prediction in zip(model_predictions, loaded_model_predictions):
                assert_allclose(model_prediction, loaded_prediction)

        # We should get the same result if we index the data from the original model and the loaded
        # model.
        _, indexed_validation_arrays = loaded_model.load_data_arrays(model.validation_files)
        if model._uses_data_generators():
            # As above, we'll just pass on this.
            pass
        else:
            model_predictions = model.model.predict(model.validation_arrays[0])
            loaded_model_predictions = loaded_model.model.predict(indexed_validation_arrays[0])

            for model_prediction, loaded_prediction in zip(model_predictions, loaded_model_predictions):
                assert_allclose(model_prediction, loaded_prediction)
        return model, loaded_model

    @staticmethod
    def one_hot(index, length):
        vector = numpy.zeros(length)
        vector[index] = 1
        return vector

    def write_snli_files(self):
        with codecs.open(self.TRAIN_FILE, 'w', 'utf-8') as train_file:
            train_file.write('1\ttext 1\thypothesis1\tentails\n')
            train_file.write('2\ttext 2\thypothesis2\tcontradicts\n')
            train_file.write('3\ttext3\thypothesis3\tentails\n')
            train_file.write('4\ttext 4\thypothesis4\tneutral\n')
            train_file.write('5\ttext5\thypothesis 5\tentails\n')
            train_file.write('6\ttext6\thypothesis6\tcontradicts\n')
        with codecs.open(self.VALIDATION_FILE, 'w', 'utf-8') as validation_file:
            validation_file.write('1\ttext 1 with extra words\thypothesis1\tentails\n')
            validation_file.write('2\ttext 2\tlonger hypothesis 2\tcontradicts\n')
            validation_file.write('3\ttext3\thypothesis withreallylongfakeword\tentails\n')

    def write_sequence_tagging_files(self):
        with codecs.open(self.TRAIN_FILE, 'w', 'utf-8') as train_file:
            train_file.write('cats###N\tare###V\tanimals###N\t.###N\n')
            train_file.write('dogs###N\tare###V\tanimals###N\t.###N\n')
            train_file.write('snakes###N\tare###V\tanimals###N\t.###N\n')
            train_file.write('birds###N\tare###V\tanimals###N\t.###N\n')
        with codecs.open(self.VALIDATION_FILE, 'w', 'utf-8') as validation_file:
            validation_file.write('horses###N\tare###V\tanimals###N\t.###N\n')
            validation_file.write('blue###N\tcows###N\tare###V\tanimals###N\t.###N\n')
            validation_file.write('monkeys###N\tare###V\tanimals###N\t.###N\n')
            validation_file.write('caterpillars###N\tare###V\tanimals###N\t.###N\n')

    def write_verb_semantics_files(self):
        with codecs.open(self.TRAIN_FILE, 'w', 'utf-8') as train_file:
            train_file.write('root####absorb####water\t1,1\t2,2\tMOVE\t-1,-1\t0,0\n')
            train_file.write('this####mixture####is####converted####into####sugar####inside####leaf'
                             '\t2,3\t5,5\tCREATE\t7,7\t-1,-1\n')
            train_file.write('lakes####contain####water\t1,1\t2,2\tNONE\t-1,-1\t-1,-1\n')
        with codecs.open(self.VALIDATION_FILE, 'w', 'utf-8') as validation_file:
            validation_file.write('root####absorb####water\t1,1\t2,2\tMOVE\t-1,-1\t0,0\n')
            validation_file.write('this####mixture####is####converted####into####sugar####inside####leaf'
                                  '\t2,3\t5,5\tCREATE\t7,7\t-1,-1\n')
            validation_file.write('lakes####contain####water\t1,1\t2,2\tNONE\t-1,-1\t-1,-1\n')

    def write_true_false_model_files(self):
        with codecs.open(self.VALIDATION_FILE, 'w', 'utf-8') as validation_file:
            validation_file.write('1\tq1a1\t0\n')
            validation_file.write('2\tq1a2\t1\n')
            validation_file.write('3\tq1a3\t0\n')
            validation_file.write('4\tq1a4\t0\n')
            validation_file.write('5\tq2a1\t0\n')
            validation_file.write('6\tq2a2\t0\n')
            validation_file.write('7\tq2a3\t1\n')
            validation_file.write('8\tq2a4\t0\n')
            validation_file.write('9\tq3a1\t0\n')
            validation_file.write('10\tq3a2\t0\n')
            validation_file.write('11\tq3a3\t0\n')
            validation_file.write('12\tq3a4\t1\n')
        with codecs.open(self.TRAIN_FILE, 'w', 'utf-8') as train_file:
            train_file.write('1\tsentence1\t0\n')
            train_file.write('2\tsentence2 word2 word3\t1\n')
            train_file.write('3\tsentence3 word2\t0\n')
            train_file.write('4\tsentence4\t1\n')
            train_file.write('5\tsentence5\t0\n')
            train_file.write('6\tsentence6\t0\n')
        with codecs.open(self.TEST_FILE, 'w', 'utf-8') as test_file:
            test_file.write('1\ttestsentence1\t0\n')
            test_file.write('2\ttestsentence2 word2 word3\t1\n')
            test_file.write('3\ttestsentence3 word2\t0\n')
            test_file.write('4\ttestsentence4\t1\n')
            test_file.write('5\ttestsentence5 word4\t0\n')
            test_file.write('6\ttestsentence6\t0\n')

    def write_additional_true_false_model_files(self):
        with codecs.open(self.VALIDATION_FILE, 'w', 'utf-8') as validation_file:
            validation_file.write('1\tq4a1\t0\n')
            validation_file.write('2\tq4a2\t1\n')
            validation_file.write('3\tq4a3\t0\n')
            validation_file.write('4\tq4a4\t0\n')
            validation_file.write('5\tq5a1\t0\n')
            validation_file.write('6\tq5a2\t0\n')
            validation_file.write('7\tq5a3\t1\n')
            validation_file.write('8\tq5a4\t0\n')
            validation_file.write('9\tq6a1\t0\n')
            validation_file.write('10\tq6a2\t0\n')
            validation_file.write('11\tq6a3\t0\n')
            validation_file.write('12\tq6a4\t1\n')
        with codecs.open(self.TRAIN_FILE, 'w', 'utf-8') as train_file:
            train_file.write('1\tsentence7\t0\n')
            train_file.write('2\tsentence8 word4 word5\t1\n')
            train_file.write('3\tsentence9 word4\t0\n')
            train_file.write('4\tsentence10\t1\n')
            train_file.write('5\tsentence11 word3 word2\t0\n')
            train_file.write('6\tsentence12\t0\n')

    def write_question_answer_files(self):
        with codecs.open(self.VALIDATION_FILE, 'w', 'utf-8') as validation_file:
            validation_file.write('1\tquestion1\tanswer1###answer2\t0\n')
        with codecs.open(self.VALIDATION_BACKGROUND, 'w', 'utf-8') as validation_background:
            validation_background.write('1\tvb1\tvb2\n')
        with codecs.open(self.TRAIN_FILE, 'w', 'utf-8') as train_file:
            train_file.write('1\ta b e i d\tanswer 1###answer2\t0\n')
            train_file.write('2\ta b c d\tanswer3###answer4\t1\n')
            train_file.write('3\te d w f d s a b\tanswer5###answer6###answer9\t2\n')
            train_file.write('4\te fj k w q\tanswer7###answer8\t0\n')
        with codecs.open(self.TRAIN_BACKGROUND, 'w', 'utf-8') as train_background:
            train_background.write('1\tsb1\tsb2\n')
            train_background.write('2\tsb3\n')
            train_background.write('3\tsb4\n')
            train_background.write('4\tsb5\tsb6\n')

    def write_who_did_what_files(self):
        with codecs.open(self.VALIDATION_FILE, 'w', 'utf-8') as validation_file:
            validation_file.write('1\tHe went to the store to buy goods, because he wanted to.'
                                  '\tHe bought xxxxx\tgoods###store\t0\n')
            validation_file.write('1\tShe hiking on the weekend with her friend.'
                                  '\tShe went xxxxx\thiking###friend###weekend###her friend\t0\n')
        with codecs.open(self.TRAIN_FILE, 'w', 'utf-8') as train_file:
            # document, question, answers
            train_file.write('1\tFred hit the ball with the bat.\tHe hit the ball with the xxxxx\tbat###ball\t0\n')
            train_file.write('1\tShe walked the dog today.\tThe xxxxx was walked today.\tShe###dog###today\t1\n')
            train_file.write('1\tHe kept typing at his desk.\tHe typed at  his xxxxx\tdesk###kept\t0\n')
            train_file.write('1\tThe pup at the bone but not the biscuit.\tThe pup ate the xxxxx\t'
                             'bone###biscuit\t0\n')

    def write_tuple_inference_files(self):
        with codecs.open(self.VALIDATION_FILE, 'w', 'utf-8') as validation_file:
            validation_file.write('1\tss<>v f d<>oo o<>c$$$s<>v ff<>o i###ss r<>v<>o e<>o ee\t'
                                  'ss ss<>ve gg<>o sd<>ccs\t0\n')
        with codecs.open(self.TRAIN_FILE, 'w', 'utf-8') as train_file:
            # document, question, answers
            train_file.write('1\tss<>v<>oo o<>c$$$s e<>ff<>o ii i###ss r<>rr<>o e<>o ee\t'
                             'ss<>ve gg<>o sd<>ccs\t0\n')
            train_file.write('2\tsg g<>vg<>oo o<>c$$$s e<>v ff<>o ii i###ss<>v rr<>o e<>o ee'
                             '###hh kk<>hdj d<>hh\tss ss<>ve gg<>o sd<>ccs\t2\n')
            train_file.write('3\ts r<>v f d<>o ss<>c$$$s e<>v ff<>o ss i$$$r<>v ss<>s o e<>o ee\t'
                             'ss ss<>v g<>o sd<>ccs\t0\n')
            train_file.write('4\tty y<>cf fv ss<>s ss<>c$$$rt e<>vv f<>oss i i###ss<>v<>os e<>o ee\t'
                             'ss ss<>ve gg<>o sd<>ccs\t1\n')

    def write_span_prediction_files(self):
        with codecs.open(self.VALIDATION_FILE, 'w', 'utf-8') as validation_file:
            validation_file.write('1\tquestion 1 with extra words\t'
                                  'passage with answer and a reallylongword\t13,18\n')
        with codecs.open(self.TRAIN_FILE, 'w', 'utf-8') as train_file:
            train_file.write('1\tquestion 1\tpassage1 with answer1\t14,20\n')
            train_file.write('2\tquestion 2\tpassage2 with answer2\t0,8\n')
            train_file.write('3\tquestion 3\tpassage3 with answer3\t9,13\n')
            train_file.write('4\tquestion 4\tpassage4 with answer4\t14,20\n')

    def write_sentence_selection_files(self):
        with codecs.open(self.VALIDATION_FILE, 'w', 'utf-8') as validation_file:
            validation_file.write('1\tWhere is Paris?\tParis is the capital of France.###It '
                                  'is by the Seine.###It is quite old###this is a '
                                  'very long sentence meant to test that loading '
                                  'and padding works properly in the model.\t1\n')
        with codecs.open(self.TRAIN_FILE, 'w', 'utf-8') as train_file:
            train_file.write('1\tWho won Super Bowl 50?\tSuper Bowl 50 was in Santa '
                             'Clara.###The Patriots beat the Broncos.\t1\n')
            train_file.write('2\tWhen is Thanksgiving?\tFolk tales tell '
                             'of the Pilgrims celebrating the holiday.###Many '
                             'people eat a lot.###It is in November.\t2\n')
            train_file.write('3\tWhen were computers invented?\tThe ancient Chinese used '
                             'abacuses.###Alan Turing cracked Enigma.###It is hard to '
                             'pinpoint an inventor of the computer.\t2\n')

    def write_pretrained_vector_files(self):
        # write the file
        with codecs.open(self.PRETRAINED_VECTORS_FILE, 'w', 'utf-8') as vector_file:
            vector_file.write('word2 0.21 0.57 0.51 0.31\n')
            vector_file.write('sentence1 0.81 0.48 0.19 0.47\n')
        # compress the file
        with open(self.PRETRAINED_VECTORS_FILE, 'rb') as f_in:
            with gzip.open(self.PRETRAINED_VECTORS_GZIP, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    def write_sentence_data(self):
        with codecs.open(self.TRAIN_FILE, 'w', 'utf-8') as train_file:
            train_file.write("This is a sentence for language modelling.\n")
            train_file.write("Here's another one for language modelling.\n")

    def write_original_snli_data(self):
        with codecs.open(self.TRAIN_FILE, 'w', 'utf-8') as train_file:
            # pylint: disable=line-too-long
            train_file.write("""{"annotator_labels": ["neutral"],"captionID": "3416050480.jpg#4", "gold_label": "neutral", "pairID": "3416050480.jpg#4r1n", "sentence1": "A person on a horse jumps over a broken down airplane.", "sentence1_binary_parse": "( ( ( A person ) ( on ( a horse ) ) ) ( ( jumps ( over ( a ( broken ( down airplane ) ) ) ) ) . ) )", "sentence1_parse": "(ROOT (S (NP (NP (DT A) (NN person)) (PP (IN on) (NP (DT a) (NN horse)))) (VP (VBZ jumps) (PP (IN over) (NP (DT a) (JJ broken) (JJ down) (NN airplane)))) (. .)))", "sentence2": "A person is training his horse for a competition.", "sentence2_binary_parse": "( ( A person ) ( ( is ( ( training ( his horse ) ) ( for ( a competition ) ) ) ) . ) )", "sentence2_parse": "(ROOT (S (NP (DT A) (NN person)) (VP (VBZ is) (VP (VBG training) (NP (PRP$ his) (NN horse)) (PP (IN for) (NP (DT a) (NN competition))))) (. .)))"}\n""")
            train_file.write("""{"annotator_labels": ["contradiction"], "captionID": "3416050480.jpg#4", "gold_label": "contradiction", "pairID": "3416050480.jpg#4r1c", "sentence1": "A person on a horse jumps over a broken down airplane.", "sentence1_binary_parse": "( ( ( A person ) ( on ( a horse ) ) ) ( ( jumps ( over ( a ( broken ( down airplane ) ) ) ) ) . ) )", "sentence1_parse": "(ROOT (S (NP (NP (DT A) (NN person)) (PP (IN on) (NP (DT a) (NN horse)))) (VP (VBZ jumps) (PP (IN over) (NP (DT a) (JJ broken) (JJ down) (NN airplane)))) (. .)))", "sentence2": "A person is at a diner, ordering an omelette.", "sentence2_binary_parse": "( ( A person ) ( ( ( ( is ( at ( a diner ) ) ) , ) ( ordering ( an omelette ) ) ) . ) )", "sentence2_parse": "(ROOT (S (NP (DT A) (NN person)) (VP (VBZ is) (PP (IN at) (NP (DT a) (NN diner))) (, ,) (S (VP (VBG ordering) (NP (DT an) (NN omelette))))) (. .)))"}\n""")
            train_file.write("""{"annotator_labels": ["entailment"], "captionID": "3416050480.jpg#4", "gold_label": "entailment", "pairID": "3416050480.jpg#4r1e", "sentence1": "A person on a horse jumps over a broken down airplane.", "sentence1_binary_parse": "( ( ( A person ) ( on ( a horse ) ) ) ( ( jumps ( over ( a ( broken ( down airplane ) ) ) ) ) . ) )", "sentence1_parse": "(ROOT (S (NP (NP (DT A) (NN person)) (PP (IN on) (NP (DT a) (NN horse)))) (VP (VBZ jumps) (PP (IN over) (NP (DT a) (JJ broken) (JJ down) (NN airplane)))) (. .)))", "sentence2": "A person is outdoors, on a horse.", "sentence2_binary_parse": "( ( A person ) ( ( ( ( is outdoors ) , ) ( on ( a horse ) ) ) . ) )", "sentence2_parse": "(ROOT (S (NP (DT A) (NN person)) (VP (VBZ is) (ADVP (RB outdoors)) (, ,) (PP (IN on) (NP (DT a) (NN horse)))) (. .)))"}\n""")
            # pylint: enable=line-too-long
        with codecs.open(self.VALIDATION_FILE, 'w', 'utf-8') as validation_file:
            # pylint: disable=line-too-long
            validation_file.write("""{"annotator_labels": ["neutral"],"captionID": "3416050480.jpg#4", "gold_label": "neutral", "pairID": "3416050480.jpg#4r1n", "sentence1": "A person on a horse jumps over a broken down airplane.", "sentence1_binary_parse": "( ( ( A person ) ( on ( a horse ) ) ) ( ( jumps ( over ( a ( broken ( down airplane ) ) ) ) ) . ) )", "sentence1_parse": "(ROOT (S (NP (NP (DT A) (NN person)) (PP (IN on) (NP (DT a) (NN horse)))) (VP (VBZ jumps) (PP (IN over) (NP (DT a) (JJ broken) (JJ down) (NN airplane)))) (. .)))", "sentence2": "A person is training his horse for a competition.", "sentence2_binary_parse": "( ( A person ) ( ( is ( ( training ( his horse ) ) ( for ( a competition ) ) ) ) . ) )", "sentence2_parse": "(ROOT (S (NP (DT A) (NN person)) (VP (VBZ is) (VP (VBG training) (NP (PRP$ his) (NN horse)) (PP (IN for) (NP (DT a) (NN competition))))) (. .)))"}\n""")
            validation_file.write("""{"annotator_labels": ["contradiction"], "captionID": "3416050480.jpg#4", "gold_label": "contradiction", "pairID": "3416050480.jpg#4r1c", "sentence1": "A person on a horse jumps over a broken down airplane.", "sentence1_binary_parse": "( ( ( A person ) ( on ( a horse ) ) ) ( ( jumps ( over ( a ( broken ( down airplane ) ) ) ) ) . ) )", "sentence1_parse": "(ROOT (S (NP (NP (DT A) (NN person)) (PP (IN on) (NP (DT a) (NN horse)))) (VP (VBZ jumps) (PP (IN over) (NP (DT a) (JJ broken) (JJ down) (NN airplane)))) (. .)))", "sentence2": "A person is at a diner, ordering an omelette.", "sentence2_binary_parse": "( ( A person ) ( ( ( ( is ( at ( a diner ) ) ) , ) ( ordering ( an omelette ) ) ) . ) )", "sentence2_parse": "(ROOT (S (NP (DT A) (NN person)) (VP (VBZ is) (PP (IN at) (NP (DT a) (NN diner))) (, ,) (S (VP (VBG ordering) (NP (DT an) (NN omelette))))) (. .)))"}\n""")
            validation_file.write("""{"annotator_labels": ["entailment"], "captionID": "3416050480.jpg#4", "gold_label": "entailment", "pairID": "3416050480.jpg#4r1e", "sentence1": "A person on a horse jumps over a broken down airplane.", "sentence1_binary_parse": "( ( ( A person ) ( on ( a horse ) ) ) ( ( jumps ( over ( a ( broken ( down airplane ) ) ) ) ) . ) )", "sentence1_parse": "(ROOT (S (NP (NP (DT A) (NN person)) (PP (IN on) (NP (DT a) (NN horse)))) (VP (VBZ jumps) (PP (IN over) (NP (DT a) (JJ broken) (JJ down) (NN airplane)))) (. .)))", "sentence2": "A person is outdoors, on a horse.", "sentence2_binary_parse": "( ( A person ) ( ( ( ( is outdoors ) , ) ( on ( a horse ) ) ) . ) )", "sentence2_parse": "(ROOT (S (NP (DT A) (NN person)) (VP (VBZ is) (ADVP (RB outdoors)) (, ,) (PP (IN on) (NP (DT a) (NN horse)))) (. .)))"}\n""")
            # pylint: enable=line-too-long
