"""
A `Sanic <http://sanic.readthedocs.io/en/latest/>`_ server that serves up
AllenNLP models as well as our demo.

Usually you would use :mod:`~allennlp.commands.serve`
rather than instantiating an ``app`` yourself.
"""
from typing import Dict
import asyncio
import json
import logging
import os
import sys
from functools import lru_cache

from sanic import Sanic, response, request
from sanic.exceptions import ServerError
from sanic_cors import CORS

from allennlp.common.util import JsonDict
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

# Can override cache size with an environment variable. If it's 0 then disable caching altogether.
CACHE_SIZE = os.environ.get("SANIC_CACHE_SIZE") or 128

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def run(port: int, workers: int,
        trained_models: Dict[str, str],
        static_dir: str = None) -> None:
    """Run the server programatically"""
    print("Starting a sanic server on port {}.".format(port))

    app = make_app(static_dir)
    CORS(app)

    for predictor_name, archive_file in trained_models.items():
        archive = load_archive(archive_file)
        predictor = Predictor.from_archive(archive, predictor_name)
        app.predictors[predictor_name] = predictor

    app.run(port=port, host="0.0.0.0", workers=workers)

def make_app(build_dir: str = None) -> Sanic:
    app = Sanic(__name__)  # pylint: disable=invalid-name

    if build_dir is None:
        # Need path to static assets to be relative to this file.
        dir_path = os.path.dirname(os.path.realpath(__file__))
        build_dir = os.path.join(dir_path, '../../demo/build')

    if not os.path.exists(build_dir):
        logger.error("app directory %s does not exist, aborting", build_dir)
        sys.exit(-1)

    app.predictors = {}

    try:
        cache_size = int(CACHE_SIZE)  # type: ignore
    except ValueError:
        logger.warning("unable to parse cache size %s as int, disabling cache", CACHE_SIZE)
        cache_size = 0

    @lru_cache(maxsize=cache_size)
    def _caching_prediction(model: Predictor, data: str) -> JsonDict:
        """
        Just a wrapper around ``model.predict_json`` that allows us to use a cache decorator.
        """
        return model.predict_json(json.loads(data))

    @app.route('/permalink/<slug>', methods=['GET'])
    async def permalink(req: request.Request, slug: str) -> response.HTTPResponse: # pylint: disable=unused-argument,unused-variable
        """
        Just return the index.html page
        """
        print(slug)
        return await response.file(os.path.join(build_dir, 'index.html'))

    @app.route('/permadata', methods=['POST', 'OPTIONS'])
    async def permadata(req: request.Request) -> response.HTTPResponse: # pylint: disable=unused-variable
        """
        Just return the index.html page
        """
        if req.method == "OPTIONS":
            return response.text("")

        slug = req.json["slug"]
        model_name = slug

        # TODO(joelgrus): don't use precanned data
        if model_name == "srl":
            request_data = {"sentence":"If you liked the music we were playing last night, you will absolutely love what we're playing tomorrow!"}
            response_data = {"words":["If","you","liked","the","music","we","were","playing","last","night",",","you","will","absolutely","love","what","we","'re","playing","tomorrow","!"],"verbs":[{"verb":"liked","description":"If [ARG0: you] [V: liked] [ARG1: the music we were playing last night] , you will absolutely love what we 're playing tomorrow !","tags":["O","B-ARG0","B-V","B-ARG1","I-ARG1","I-ARG1","I-ARG1","I-ARG1","I-ARG1","I-ARG1","O","O","O","O","O","O","O","O","O","O","O"]},{"verb":"were","description":"If you liked the music we [V: were] playing last night , you will absolutely love what we 're playing tomorrow !","tags":["O","O","O","O","O","O","B-V","O","O","O","O","O","O","O","O","O","O","O","O","O","O"]},{"verb":"playing","description":"If you liked [ARG1: the music] [ARG0: we] were [V: playing] [ARGM-TMP: last night] , you will absolutely love what we 're playing tomorrow !","tags":["O","O","O","B-ARG1","I-ARG1","B-ARG0","O","B-V","B-ARGM-TMP","I-ARGM-TMP","O","O","O","O","O","O","O","O","O","O","O"]},{"verb":"will","description":"[ARGM-ADV: If you liked the music we were playing last night] , [ARG0: you] [V: will] [ARG1: absolutely love what we 're playing tomorrow] !","tags":["B-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","O","B-ARG0","B-V","B-ARG1","I-ARG1","I-ARG1","I-ARG1","I-ARG1","I-ARG1","I-ARG1","O"]},{"verb":"love","description":"[ARGM-ADV: If you liked the music we were playing last night] , [ARG0: you] [ARGM-MOD: will] [ARGM-ADV: absolutely] [V: love] [ARG1: what we 're playing tomorrow] !","tags":["B-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","O","B-ARG0","B-ARGM-MOD","B-ARGM-ADV","B-V","B-ARG1","I-ARG1","I-ARG1","I-ARG1","I-ARG1","O"]},{"verb":"'re","description":"If you liked the music we were playing last night , you will absolutely love what we [V: 're] playing tomorrow !","tags":["O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","B-V","O","O","O"]},{"verb":"playing","description":"If you liked the music we were playing last night , you will absolutely love [ARG1: what] [ARG0: we] 're [V: playing] [ARGM-TMP: tomorrow] !","tags":["O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","B-ARG1","B-ARG0","O","B-V","B-ARGM-TMP","O"]}],"tokens":["If","you","liked","the","music","we","were","playing","last","night",",","you","will","absolutely","love","what","we","'re","playing","tomorrow","!"]}
        elif model_name == "mc":
            request_data = {"passage":"A reusable launch system (RLS, or reusable launch vehicle, RLV) is a launch system which is capable of launching a payload into space more than once. This contrasts with expendable launch systems, where each launch vehicle is launched once and then discarded. No completely reusable orbital launch system has ever been created. Two partially reusable launch systems were developed, the Space Shuttle and Falcon 9. The Space Shuttle was partially reusable: the orbiter (which included the Space Shuttle main engines and the Orbital Maneuvering System engines), and the two solid rocket boosters were reused after several months of refitting work for each launch. The external tank was discarded after each flight.","question":"How many partially reusable launch systems were developed?"}
            response_data = {"span_start_logits":[-1.9083371162,-4.2220363617,-6.9773244858,-8.4537620544,-8.0504684448,-5.7489509583,-10.2273607254,-10.9374141693,-6.5089511871,-8.353272438,-7.6484603882,-11.3071727753,-6.4416899681,-10.0333652496,-8.3851795197,-6.958776474,-7.1654763222,-9.1572856903,-9.0032367706,-8.6910171509,-7.2494492531,-10.1062097549,-6.9719047546,-6.7065753937,-6.6466760635,-7.1442260742,-4.6959791183,0.8482889533,-3.5672237873,-3.9982578754,-7.1446213722,-6.5219984055,-5.6457486153,-5.7085895538,-3.5556793213,-6.2677602768,-8.2498579025,-7.9215006828,-3.6418249607,-3.1652641296,-5.4989037514,-5.9583826065,-6.2445449829,-5.1549444199,-2.4177210331,-7.816775322,-4.0728878975,-4.5569262505,-4.5557370186,2.8955967426,-3.3393187523,-5.3276834488,-6.5457119942,-6.3199381828,-7.6391506195,-5.5430526733,-2.938202858,-6.2751402855,-3.6240484715,-1.5142711401,8.5821313858,-3.6672592163,-5.2849626541,-6.0935544968,-7.4137997627,-7.7574944496,-6.4922885895,-7.6087818146,-5.0685720444,-5.9570274353,-8.3002309799,-10.5016860962,-5.147646904,-4.2090549469,-6.3657536507,-1.8135653734,-2.2449655533,-6.143219471,-8.8654193878,-7.7030687332,-9.4098186493,-11.9053068161,-8.9177465439,-8.1304712296,-10.3954496384,-11.358291626,-9.9891195297,-9.7995300293,-9.2956724167,-11.1642665863,-10.559715271,-10.1894540787,-13.6026678085,-10.5481491089,-9.9993743896,-10.9570789337,-11.429772377,-10.1077451706,-12.609085083,-10.6987991333,-10.2644748688,-6.6579217911,-4.1839046478,-8.7422275543,-10.6024589539,-10.377038002,-12.1914539337,-10.4462490082,-10.1870145798,-9.2385091782,-9.8182106018,-13.2289457321,-9.6628904343,-11.599111557,-11.8057479858,-11.3020391464,-13.6569757462,-13.3220663071,-8.560956955,-10.3444385529,-11.8461961746,-13.5682621002,-11.0669937134,-11.6203536987,-12.9780092239,-12.1475219727,-10.2725162506],"span_start_probs":[0.0000276879,0.0000027382,0.0000001741,0.0000000398,0.0000000595,0.0000005947,0.0000000068,0.0000000033,0.0000002781,0.000000044,0.000000089,0.0000000023,0.0000002975,0.0000000082,0.0000000426,0.0000001774,0.0000001443,0.0000000197,0.000000023,0.0000000314,0.0000001326,0.0000000076,0.0000001751,0.0000002283,0.0000002424,0.0000001474,0.0000017046,0.000435992,0.0000052704,0.0000034249,0.0000001473,0.0000002745,0.0000006594,0.0000006192,0.0000053316,0.000000354,0.0000000488,0.0000000677,0.0000048916,0.000007878,0.0000007637,0.0000004824,0.0000003623,0.0000010772,0.0000166367,0.0000000752,0.0000031786,0.000001959,0.0000019613,0.0033776362,0.0000066195,0.0000009063,0.0000002681,0.000000336,0.0000000898,0.0000007307,0.0000098861,0.0000003514,0.0000049793,0.0000410612,0.9959638119,0.0000047687,0.0000009459,0.0000004214,0.0000001125,0.0000000798,0.0000002828,0.0000000926,0.0000011744,0.000000483,0.0000000464,0.0000000051,0.0000010851,0.000002774,0.000000321,0.0000304404,0.000019774,0.000000401,0.0000000264,0.0000000843,0.0000000153,0.0000000013,0.000000025,0.000000055,0.0000000057,0.0000000022,0.0000000086,0.0000000104,0.0000000171,0.0000000026,0.0000000048,0.000000007,0.0000000002,0.0000000049,0.0000000085,0.0000000033,0.000000002,0.0000000076,0.0000000006,0.0000000042,0.0000000065,0.0000002396,0.0000028446,0.0000000298,0.0000000046,0.0000000058,0.0000000009,0.0000000054,0.000000007,0.0000000181,0.0000000102,0.0000000003,0.0000000119,0.0000000017,0.0000000014,0.0000000023,0.0000000002,0.0000000003,0.0000000357,0.000000006,0.0000000013,0.0000000002,0.0000000029,0.0000000017,0.0000000004,0.000000001,0.0000000065],"span_end_logits":[-7.1659440994,-8.9594802856,-8.5077524185,-6.4721441269,-11.4557561874,-8.4238290787,-10.3626308441,-12.3125677109,-10.0014944077,-9.6650810242,-8.2614946365,-11.7233772278,-9.4618911743,-8.0739345551,-12.2181453705,-11.7887039185,-10.1296367645,-7.5513672829,-11.2352180481,-12.6370973587,-11.2701501846,-12.6840267181,-11.5653142929,-11.8983078003,-10.3960285187,-12.9412784576,-9.3430156708,-7.5779027939,-10.1216211319,-5.0105829239,-7.5200271606,-10.9781446457,-9.9437026978,-11.9360361099,-10.1841955185,-9.8811035156,-7.8825955391,-9.5833473206,-8.5677108765,-9.8796377182,-10.2267990112,-8.4236650467,-11.3640060425,-9.9744958878,-6.4092130661,-10.1833791733,-10.3816165924,-8.0445413589,-7.970354557,-2.8977634907,-8.1620063782,-9.0012264252,-8.2409591675,-8.5887088776,-6.8616380692,-10.8423118591,-7.5718927383,-10.4629211426,-7.8369350433,-7.0562787056,2.637083292,-5.5475769043,-7.0808649063,-7.3044548035,-5.4672436714,-10.2844629288,-3.7595975399,-5.5443134308,-8.8456697464,-9.2263202667,-6.3339018822,-10.4794197083,-8.1708011627,-3.5161612034,-4.6175847054,-9.1829776764,-9.7004413605,-5.3220691681,-11.3741340637,-10.1349124908,-8.6104345322,-9.5716304779,-11.1203842163,-7.018488884,-11.1496810913,-10.604850769,-11.267865181,-11.7460346222,-11.4954614639,-9.7916564941,-11.7466745377,-9.4757175446,-12.38671875,-12.0337524414,-10.9501409531,-10.5399446487,-11.3414983749,-9.8690729141,-8.598991394,-9.7292470932,-11.9556102753,-11.6110181808,-6.5470128059,-11.0615787506,-9.854798317,-8.729924202,-13.3746414185,-10.546918869,-11.6951789856,-11.3322925568,-9.5649280548,-12.9792566299,-10.9553432465,-9.9980363846,-12.8673686981,-11.7899188995,-9.6026821136,-9.801451683,-11.6788158417,-12.4904327393,-10.9261541367,-13.2214288712,-11.0376968384,-12.5423879623,-12.5098266602,-10.0286178589,-9.2021961212],"span_end_probs":[0.0000546487,0.000009092,0.0000142837,0.0001093688,0.0000007491,0.0000155342,0.000002235,0.000000318,0.0000032071,0.0000044897,0.0000182721,0.0000005732,0.0000055013,0.0000220417,0.0000003495,0.000000537,0.0000028214,0.00003717,0.0000009339,0.0000002299,0.0000009019,0.0000002193,0.0000006714,0.0000004812,0.0000021616,0.0000001696,0.0000061957,0.0000361967,0.0000028441,0.0004716736,0.0000383534,0.0000012077,0.0000033979,0.0000004634,0.0000026716,0.0000036174,0.0000266896,0.0000048721,0.0000134524,0.0000036227,0.0000025602,0.0000155367,0.0000008211,0.0000032949,0.0001164727,0.0000026738,0.000002193,0.0000226992,0.0000244472,0.0039014614,0.0000201835,0.0000087202,0.0000186512,0.0000131729,0.0000740864,0.0000013834,0.0000364149,0.0000020217,0.0000279365,0.0000609828,0.9885092378,0.0002756945,0.0000595017,0.0000475801,0.0002987557,0.0000024167,0.0016479254,0.0002765957,0.0000101879,0.0000069626,0.0001255831,0.0000019886,0.0000200067,0.0021021352,0.0006987444,0.000007271,0.0000043337,0.0003454338,0.0000008128,0.0000028066,0.0000128898,0.0000049295,0.0000010476,0.0000633314,0.0000010173,0.0000017542,0.0000009039,0.0000005604,0.0000007199,0.0000039559,0.00000056,0.0000054257,0.0000002953,0.0000004203,0.000001242,0.0000018718,0.0000008398,0.0000036612,0.0000130381,0.0000042107,0.0000004544,0.0000006414,0.0001014795,0.000001111,0.0000037139,0.0000114381,0.0000001099,0.0000018588,0.0000005896,0.0000008475,0.0000049627,0.0000001633,0.0000012356,0.0000032182,0.0000001826,0.0000005363,0.0000047788,0.0000039174,0.0000005993,0.0000002662,0.0000012722,0.0000001281,0.0000011379,0.0000002527,0.0000002611,0.0000031213,0.0000071326],"best_span":[60,60],"best_span_str":"Two"}
        elif model_name == "te":
            request_data = {"premise":"If you help the needy, God will reward you.","hypothesis":"Giving money to the poor has good consequences."}
            response_data = {"label_logits":[-0.0950802863,-0.5699284673,0.5117189884],"label_probs":[0.2893075049,0.179943338,0.5307491422]}

        await asyncio.sleep(0.25)

        return response.json({
                "modelName": model_name,
                "requestData": request_data,
                "responseData": response_data
        })

    @app.route('/predict/<model_name>', methods=['POST', 'OPTIONS'])
    async def predict(req: request.Request, model_name: str) -> response.HTTPResponse:  # pylint: disable=unused-variable
        """make a prediction using the specified model and return the results"""
        if req.method == "OPTIONS":
            return response.text("")

        model = app.predictors.get(model_name.lower())
        if model is None:
            raise ServerError("unknown model: {}".format(model_name), status_code=400)

        data = req.json
        log_blob = {"model": model_name, "inputs": data, "cached": False, "outputs": {}}

        # See if we hit or not. In theory this could result in false positives.
        pre_hits = _caching_prediction.cache_info().hits  # pylint: disable=no-value-for-parameter

        try:
            if cache_size > 0:
                # lru_cache insists that all function arguments be hashable,
                # so unfortunately we have to stringify the data.
                prediction = _caching_prediction(model, json.dumps(data))
            else:
                # if cache_size is 0, skip caching altogether
                prediction = model.predict_json(data)
        except KeyError as err:
            raise ServerError("Required JSON field not found: " + err.args[0], status_code=400)

        post_hits = _caching_prediction.cache_info().hits  # pylint: disable=no-value-for-parameter

        if post_hits > pre_hits:
            # Cache hit, so insert an artifical pause
            log_blob["cached"] = True
            await asyncio.sleep(0.25)

        # The model predictions are extremely verbose, so we only log the most human-readable
        # parts of them.
        if model_name == "machine-comprehension":
            log_blob["outputs"]["best_span_str"] = prediction["best_span_str"]
        elif model_name == "textual-entailment":
            log_blob["outputs"]["label_probs"] = prediction["label_probs"]
        elif model_name == "semantic-role-labeling":
            verbs = []

            for verb in prediction["verbs"]:
                # Don't want to log boring verbs with no semantic parses.
                good_tags = [tag for tag in verb["tags"] if tag != "0"]
                if len(good_tags) > 1:
                    verbs.append({"verb": verb["verb"], "description": verb["description"]})

            log_blob["outputs"]["verbs"] = verbs

        logger.info("prediction: %s", json.dumps(log_blob))

        return response.json(prediction)

    @app.route('/models')
    async def list_models(req: request.Request) -> response.HTTPResponse:  # pylint: disable=unused-argument, unused-variable
        """list the available models"""
        return response.json({"models": list(app.predictors.keys())})

    app.static('/', os.path.join(build_dir, 'index.html'))
    app.static('/', build_dir)

    return app
