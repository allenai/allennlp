# Roadmap

To better communicate with the community, we plan to share our main focuses each quarter in this
document.

# Q4 2019 (October, November, December)

We have a few major themes for this quarter:

1.  Breaking out models into their own repositories (you can follow [this issue](https://github.com/allenai/allennlp/issues/3351) for more information).  By doing so we hope to increase community engagement and ownership of these model repositories, allowing us to focus more on the core library.  We already have a good start on this with the following repositories:

    a.  [Semantic Parsing](https://github.com/allenai/allennlp-semparse)
    
    b.  [Reading Comprehension](https://github.com/allenai/allennlp-reading-comprehension)

    We will test out this idea of splitting up the repositories with these two first, fixing any issues with them by the end of this quarter.  Around the end of this quarter or early next quarter, we will finish the process with the rest of the models, making repositories around language modeling, coreference resolution, sequence tagging, parsing, etc.

2.  Better supporting our initiative around [Green AI](https://arxiv.org/abs/1907.10597). 

    a.  Improve the performance of AllenNLP.  In particular, see the following issues:
    
       1.  [Using DistributedDataParallel for multi GPU training](https://github.com/allenai/allennlp/issues/2536)
        
       2.  [Native pytorch Multiprocessing for data loading](https://github.com/allenai/allennlp/issues/3079)
        
       3.  [Mixed Precision Training](https://github.com/allenai/allennlp/issues/2149)
        
    b.  Add tooling so any user can easily measure the total number of floating-point operations
    performed during training.  You can see more details about this initiative in [#3436](https://github.com/allenai/allennlp/issues/3436)

3.  Launching a first version of an AllenNLP course.  It's challenging for many people to onboard,
    so we're building an interactive course.  We're also designing it to be an effective resource
    for people teaching introductory NLP courses.

We're also starting to think about our 1.0 release, and have started organizing issues around [the relevant milestone](https://github.com/allenai/allennlp/milestone/10).
