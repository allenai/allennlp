# pylint: disable=invalid-name,protected-access
import codecs
import gzip
import logging
import os
import sys
import shutil
from unittest import TestCase

import torch
import pytest
from numpy.testing import assert_allclose

from allennlp.common.checks import log_pytorch_version_info
from allennlp.common.params import Params
from allennlp.common.tensor import arrays_to_variables
from allennlp.data.dataset import Dataset
from allennlp.data.iterators import BasicIterator
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model


class AllenNlpTestCase(TestCase):  # pylint: disable=too-many-public-methods
    TEST_DIR = './TMP_TEST/'
    MODEL_FILE = TEST_DIR + "model.th"
    TRAIN_FILE = TEST_DIR + 'train_file'
    VALIDATION_FILE = TEST_DIR + 'validation_file'
    TEST_FILE = TEST_DIR + 'test_file'
    TRAIN_BACKGROUND = TEST_DIR + 'train_background'
    VALIDATION_BACKGROUND = TEST_DIR + 'validation_background'
    SNLI_FILE = TEST_DIR + 'snli_file'
    PRETRAINED_VECTORS_FILE = TEST_DIR + 'pretrained_glove_vectors_file'
    PRETRAINED_VECTORS_GZIP = TEST_DIR + 'pretrained_glove_vectors_file.gz'
    CONLL_TRAIN_DIR = TEST_DIR + 'conll/train/'
    CONLL_VAL_DIR = TEST_DIR + 'conll/val/'

    def setUp(self):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            level=logging.DEBUG)
        log_pytorch_version_info()
        os.makedirs(self.TEST_DIR, exist_ok=True)
        os.makedirs(self.CONLL_TRAIN_DIR + "english/annotations/test_topic/test_source/01/", exist_ok=True)
        os.makedirs(self.CONLL_VAL_DIR + "english/annotations/test_topic/test_source/01/", exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.TEST_DIR)

    @staticmethod
    @pytest.fixture(scope='module')
    def unload_registrable():
        # Each test class which tests registrable subclasses registers the
        # subclasses when they are imported into the test file, which is
        # modifying a class attribute of "Registrable". In order to test that
        # subclasses are being correctly registered for each set of registrable
        # things, we want to clear the registry in the tests between each test.
        # However, if we do this between individual tests, we clear the registry
        # prematurely, as the imports at the top of a file containing a test class
        # are not re-run for every individual test within a test class, causing some
        # tests to fail as the registry does not have the correct keys.

        # This clears the registry in between each test file (due to the 'module'
        # level scope), preventing this behaviour and # making the tests as
        # maximally isolated as feasible.

        del sys.modules['allennlp.common.registrable']

    def get_trainer_params(self, additional_arguments=None):
        params = Params({})
        params['save_models'] = False
        params['serialization_prefix'] = self.MODEL_FILE
        params['num_epochs'] = 1

        if additional_arguments:
            for key, value in additional_arguments.items():
                params[key] = value
        return params

    def ensure_model_can_train_save_and_load(self,
                                             model: Model,
                                             dataset: Dataset,
                                             iterator: DataIterator = None):
        data_iterator = iterator or BasicIterator()
        single_batch = next(data_iterator(dataset))
        single_batch = arrays_to_variables(single_batch)
        model_predictions = model.forward(**single_batch)

        # Check loss exists and we can compute gradients.
        model_loss = model_predictions["loss"]
        assert model_loss
        model_loss.backward()

        torch.save(model.state_dict(), self.MODEL_FILE)
        loaded_model = model
        loaded_model.zero_grad()
        loaded_model.load_state_dict(torch.load(self.MODEL_FILE))
        loaded_model_predictions = loaded_model.forward(**single_batch)

        # Check loaded model's loss exists and we can compute gradients.
        loaded_model_loss = loaded_model_predictions["loss"]
        assert loaded_model_loss
        loaded_model_loss.backward()

        # Both outputs should have the same keys and the values
        # for these keys should be close.
        for key in model_predictions.keys():
            assert_allclose(model_predictions[key].data.numpy(), loaded_model_predictions[key].data.numpy())

        return model, loaded_model

    def write_pretrained_vector_files(self):
        # write the file
        with codecs.open(self.PRETRAINED_VECTORS_FILE, 'w', 'utf-8') as vector_file:
            vector_file.write('word2 0.21 0.57 0.51 0.31\n')
            vector_file.write('sentence1 0.81 0.48 0.19 0.47\n')
        # compress the file
        with open(self.PRETRAINED_VECTORS_FILE, 'rb') as f_in:
            with gzip.open(self.PRETRAINED_VECTORS_GZIP, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    def write_sequence_tagging_data(self):
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

    def write_sentence_data(self):
        with codecs.open(self.TRAIN_FILE, 'w', 'utf-8') as train_file:
            train_file.write("This is a sentence for language modelling.\n")
            train_file.write("Here's another one for language modelling.\n")

    def write_snli_data(self):
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

    def write_conll_2012_data(self):
        # pylint: disable=line-too-long
        with codecs.open(self.CONLL_TRAIN_DIR + 'english/annotations/test_topic/test_source/01/train.gold_conll', 'w', 'utf-8') as train_file:
            train_file.write("""#begin document (bn/cnn/01/cnn_0115); part 000
test/test/01/test_001  0    0          Mali   NNP  (TOP(S(NP(NML*          -    -   -   -    (GPE)  (ARG0*         *       -
test/test/01/test_001  0    1    government    NN               *)         -    -   -   -       *        *         *       -
test/test/01/test_001  0    2     officials   NNS               *)   official   -   1   -       *        *)        *       -
test/test/01/test_001  0    3           say   VBP            (VP*         say  01   1   -       *      (V*)        *       -
test/test/01/test_001  0    4           the    DT  (SBAR(S(NP(NP*          -    -   -   -       *   (ARG1*    (ARG1*   (1|(3
test/test/01/test_001  0    5         woman    NN               *         man   -   2   -       *        *         *       -
test/test/01/test_001  0    6            's   POS               *)         -    -   -   -       *        *         *       1)
test/test/01/test_001  0    7    confession    NN               *)         -    -   -   -       *        *         *)      3)
test/test/01/test_001  0    8           was   VBD            (VP*          be  01   1   -       *        *       (V*)      -
test/test/01/test_001  0    9        forced    JJ      (ADJP*)))))         -    -   -   -       *        *)   (ARG2*)      -
test/test/01/test_001  0   10             .     .              *))         -    -   -   -       *        *         *       -

test/test/02/test_002  0    0            The     DT  (TOP(S(NP*             -    -   -   -        *       (ARG0*      *    (2
test/test/02/test_002  0    1    prosecution     NN           *)   prosecution   -   2   -        *            *)     *     2)
test/test/02/test_002  0    2         rested    VBD        (VP*           rest  01   5   -        *          (V*)     *     -
test/test/02/test_002  0    3            its   PRP$        (NP*             -    -   -   -        *       (ARG1*      *    (2)
test/test/02/test_002  0    4           case     NN           *)          case   -   2   -        *            *)     *     -
test/test/02/test_002  0    5           last     JJ        (NP*             -    -   -   -   (DATE*   (ARGM-TMP*      *     -
test/test/02/test_002  0    6          month     NN           *)            -    -   -   -        *)           *)     *     -
test/test/02/test_002  0    7          after     IN        (PP*             -    -   -   -        *   (ARGM-TMP*      *     -
test/test/02/test_002  0    8           four     CD     (NP(NP*             -    -   -   -   (DATE*            *      *     -
test/test/02/test_002  0    9         months    NNS           *)         month   -   1   -        *)           *      *     -
test/test/02/test_002  0   10             of     IN        (PP*             -    -   -   -        *            *      *     -
test/test/02/test_002  0   11       hearings    NNS    (NP*)))))       hearing  01   1   -        *            *)   (V*)    -
test/test/02/test_002  0   12              .      .          *))            -    -   -   -        *            *      *     -

test/test/03/test_003  0   0      Denise   NNP  (TOP(FRAG(NP*   -   -   -   -        (PERSON*   (2
test/test/03/test_003  0   1      Dillon   NNP              *)  -   -   -   -               *)   2)
test/test/03/test_003  0   4    Headline   NNP           (NP*   -   -   -   -   (WORK_OF_ART*    -
test/test/03/test_003  0   5        News   NNP              *)  -   -   -   -               *)   -
test/test/03/test_003  0   7           .     .             *))  -   -   -   -               *    -

#end document
            """)

        with codecs.open(self.CONLL_VAL_DIR + 'english/annotations/test_topic/test_source/01/val.gold_conll', 'w', 'utf-8') as validation_file:
            validation_file.write("""#begin document (bn/cnn/01/cnn_0115); part 000
test/test/01/test_001  0   0      Denise   NNP  (TOP(FRAG(NP*   -   -   -   -        (PERSON*   (2
test/test/01/test_001  0   1      Dillon   NNP              *)  -   -   -   -               *)   2)
test/test/01/test_001  0   2           ,     ,              *   -   -   -   -               *    -
test/test/01/test_001  0   3          ``    ``              *   -   -   -   -               *    -
test/test/01/test_001  0   4    Headline   NNP           (NP*   -   -   -   -   (WORK_OF_ART*    -
test/test/01/test_001  0   5        News   NNP              *)  -   -   -   -               *)   -
test/test/01/test_001  0   6           .     .              *   -   -   -   -               *    -
test/test/01/test_001  0   7          ''    ''             *))  -   -   -   -               *    -

test/test/01/test_002  0    0           The    DT   (TOP(S(NP*         -    -   -   -            *   (ARG1*          *      *        *   (0
test/test/01/test_002  0    1         three    CD            *         -    -   -   -    (CARDINAL)       *          *      *        *    -
test/test/01/test_002  0    2    defendants   NNS            *)        -    -   -   -            *        *)         *      *        *    0)
test/test/01/test_002  0    3           are   VBP         (VP*         be  01   2   -            *      (V*)         *      *        *    -
test/test/01/test_002  0    4         among    IN         (PP*         -    -   -   -            *   (ARG2*          *      *        *    -
test/test/01/test_002  0    5           27     CD      (NP(NP*         -    -   -   -    (CARDINAL)       *          *      *   (ARG1*    -
test/test/01/test_002  0    6      suspects   NNS            *)        -    -   -   -            *        *          *      *        *)   -
test/test/01/test_002  0    7      believed   VBN         (VP*    believe  01   1   -            *        *        (V*)     *        *    -
test/test/01/test_002  0    8            to    TO       (S(VP*         -    -   -   -            *        *   (R-ARG1*      *        *    -
test/test/01/test_002  0    9            be    VB         (VP*         be  03   -   -            *        *          *    (V*)       *    -
test/test/01/test_002  0   10      involved   VBN         (VP*    involve  01   1   -            *        *          *      *      (V*)   -
test/test/01/test_002  0   11            in    IN         (PP*         -    -   -   -            *        *          *      *   (ARG2*    -
test/test/01/test_002  0   12           the    DT         (NP*         -    -   -   -            *        *          *      *        *   (1
test/test/01/test_002  0   13      bombings   NNS   *))))))))))        -    -   -   -            *        *)         *)     *        *)   1)
test/test/01/test_002  0   14             .     .           *))        -    -   -   -            *        *          *      *        *    -

#end document
            """)
# pylint: enable=line-too-long
