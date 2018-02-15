
# Visualizing model internals in a live demo (BETA)

We recently added an attention visualization to the live demo on
[allennlp.org](http://demo.allennlp.org/machine-comprehension).  Here's what it looks like:

![demo screenshot](visualization_images/bidaf_attention_demo.png)

The basic components here are reusable, so you can relatively easily get a similar demo for
whatever model you're working on.  This is still pretty rough, but if you're feeling adventurous,
here's a guide to the changes you need to make to get a live demo that shows model internals, which
we've found to be very helpful for understanding what parts of our models need fixing.

## Pre-requisites

You need to have a demo of your model already running.  See the [tutorial on creating a
demo](../getting_started/making_predictions_and_creating_a_demo.md) for how to do this.  You'll
also have to be using react to serve the demo, which means the simple server is out.  You need to
be using `python -m allennlp.run serve` instead of `python -m allennlp.service.server_simple` to
run the server.  Sorry, but we're not including instructions for that here.  You can see
instructions for running the main demo in the [demo README](../../demo/README.md), but you'll have
to figure out how to either get the same thing running with your stand-alone demo or add your model
to what's already there.  I said this was still rough.

You should also be at least vaguely familiar with [react](https://reactjs.org) and how JSX
components work.

## Model plumbing

You need the model to output whatever information you want to display.  We already return a
dictionary from `Model.forward` containing whatever keys you want; you just need to be sure the
things you want to visualize are in there (including actual strings instead of integers if you want
to have human-readable labels for the rows and columns of your visualizations).  For BiDAF, this
was straightforward (see https://github.com/allenai/allennlp/pull/692/files):

- We added `question_tokens` and `passage_tokens` fields to the metadata returned for each
  instance in the `DatasetReader`, which we use for column and row labels.
- We added the `passage_question_attention` tensor to the model's `output_dict`.  No change or
  anything required here, just add the tensor directly, and the library will sanitize it to a JSON
object containing lists of numbers for you.
- In `Model.decode` we made sure that the `question_tokens` and `passage_tokens` fields got passed
  along to the `output_dict` from the `metadata` we got from the `DatasetReader`.  There are other
ways of doing this without going through a `metadata` field (e.g., change the `Predictor` to
tokenize your data, and pass along the tokens there), but it was easiest this way with BiDAF.

That's all you have to change in the model.  All we need is to be able to get the right information
in the output dictionary from `model.forward()`, however you want to make that happen.

## UI plumbing

We've added two components to the demo library that are re-usable for any model internals you want
to show: `Collapsible` and `HeatMap`.  You just need to get the attention values you returned from
`model.forward()` passed in to a `HeatMap` component with the right labels.  Again, with BiDAF,
this was pretty straightforward.  Here we're modifying `McComponent.js`, because that's where the
BiDAF rendering components are found (again see
https://github.com/allenai/allennlp/pull/692/files):

- We found the place where the request data was read and grabbed another field from the
  returned json.  This looks something like `const attention = responseData &&
responseData.passage_question_attention`, and similar for the `question_tokens` and
`passage_tokens`.
- We passed those fields in to the `McOutput` component, which renders the model output.
- We modified the `McOutput` component to accept and render those fields.  We added two
  `Collapsible` components, one wrapping all model internals, and one containing the attention
heatmap.  The `HeatMap` component just takes the `question_tokens` as `xLabels`, the
`passage_tokens` as `yLabels`, and the `attention` as `data`.  You can see the specific code in
the linked pull request above.

That should be all you have to do to get model internals visualized in your demo.

## Single-dimension heatmaps

The `HeatMap` component we used expects a two-dimensional data array.  If you have a
one-dimensional data array you want to visualize, this is still possible, but you need to do
something like `attention = attention.map(x => [x])` in the javascript to make your
one-dimensional array into a two-dimensional array with length 1 on the x axis.  You also need to
provide a single `xLabel` to the `HeatMap`, like `xLabels={['Probabilities']}`.

## More general uses

The basic outline here is quite general and allows for visualizing all kinds of model internals.
You just need to make sure your model outputs the internal state you want to visualize, and that
the demo code passes that information from the JSON response to the proper react component.

We've used this to visualize the internal workings of an action-based semantic parser.  The parser
is not quite ready for public consumption, so the code and demo are not yet available, but here's
a sneak peak of what you can do with this kind of model visualization:

|![main parser UI](visualization_images/wikitables_overview.png)|
|:--:|
| *The main (still rough) parser UI, with model internals* |

|<img src="visualization_images/predicted_actions.png" width="400" />|
|:--:|
| *The sequence of predicted actions that generated the logical form above* |

|<img src="visualization_images/action_detail.png" width="300" />|
|:--:|
| *At each output step, we show the considered actions and their probabilities* |

|<img src="visualization_images/action_detail_2.png" width="150" />|
|:--:|
| *At each output step, we also show the model's attention over the question* |

|![linking scores](visualization_images/linking_scores.png)|
|:--:|
| *Before decoding, we compute a linking between table entities and question words. This shows part of that linking.* |

