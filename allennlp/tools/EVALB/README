#################################################################
#                                                               #
#      Bug fix and additional functionality for evalb           #
#                                                               #
# This updated version of evalb fixes a bug in which sentences  #
# were incorrectly categorized as "length mismatch" when the    #
# the parse output had certain mislabeled parts-of-speech.      #
#                                                               #
# The bug was the result of evalb treating one of the tags (in  #
# gold or test) as a label to be deleted (see sections [6],[7]  #
# for details), but not the corresponding tag in the other.     #
# This most often occurs with punctuation. See the subdir       #
# "bug" for an example gld and tst file demonstating the bug,   #
# as well as output of evalb with and without the bug fix.      #
#                                                               #
# For the present version in case of length mismatch, the nodes #
# causing the imbalance are reinserted to resolve the miscount. #
# If the lengths of gold and test truly differ, the error is    #
# still reported. The parameter file "new.prm" (derived from    #
# COLLINS.prm) shows how to add new potential mislabelings for  #
# quotes (",``,',`).                                            #
#                                                               #
# I have preserved DJB's revision for modern compilers except   #
# for the delcaration of "exit" which is provided by stdlib.    #
#                                                               #
# Other changes:                                                #
#                                                               #
# * output of F-Measure in addition to precision and recall     #
#   (I did not update the documention in section [4] for this)  #
#                                                               #
# * more comprehensive DEBUG output that includes bracketing    #
#   information as evalb is processing each sentence            #
#   (useful in working through this, and peraps other bugs).    #
#   Use either the "-D" run-time switch or set DEBUG to 2 in    #
#   the parameter file.                                         #
#                                                               #
# * added DELETE_LABEL lines in new.prm for S1 nodes produced   #
#   by the Charniak parser and "?", "!" punctuation produced by #
#   the Bikel parser.                                           #
#                                                               #
#                                                               #
#                                           David Ellis (Brown) #
#                                                               #
#                                           January.2006        #
#################################################################

#################################################################
#                                                               #
#      Update of evalb for modern compilers                     #
#                                                               #
# This is an updated version of evalb, for use with modern C    #
# compilers. There are a few updates, each marked in the code:  #
#                                                               #
# /* DJB: explanation of comment */                             #
#                                                               #
# The updates are purely to help compilation with recent        #
# versions of GCC (and other C compilers). There are *NO* other #
# changes to the algorithm itself.                              #
#                                                               #
# I have made these changes following recommendations from      #
# users of the Corpora Mailing List, especially Peet Morris and #
# Ramon Ziai.                                                   #
#                                                               #
#                                     David Brooks (Birmingham) #
#                                                               #
#                                     September.2005            #
#################################################################

#################################################################
#                                                               #
#      README file for evalb                                    #
#                                                               #
#                                         Satoshi Sekine (NYU)  #
#                                         Mike Collins (UPenn)  #
#                                                               #
#                                         October.1997          #
#################################################################

Contents of this README:

   [0] COPYRIGHT
   [1] INTRODUCTION
   [2] INSTALLATION AND RUN
   [3] OPTIONS
   [4] OUTPUT FORMAT FROM THE SCORER
   [5] HOW TO CREATE A GOLDFILE FROM THE TREEBANK
   [6] THE PARAMETER FILE
   [7] MORE DETAILS ABOUT THE SCORING ALGORITHM


[0] COPYRIGHT

The authors abandon the copyright of this program. Everyone is 
permitted to copy and distribute the program or a portion of the program
with no charge and no restrictions unless it is harmful to someone.

However, the authors are delightful for the user's kindness of proper
usage and letting the authors know bugs or problems.

This software is provided "AS IS", and the authors make no warranties,
express or implied.

To legally enforce the abandonment of copyright, this package is released
under the Unlicense (see LICENSE).

[1] INTRODUCTION

Evaluation of bracketing looks simple, but in fact, there are minor
differences from system to system. This is a program to parametarize
such minor differences and to give an informative result.

"evalb" evaluates bracketing accuracy in a test-file against a gold-file.
It returns recall, precision, tagging accuracy. It uses an identical 
algorithm to that used in (Collins ACL97).


[2] Installation and Run

To compile the scorer, type 

> make


To run the scorer:

> evalb -p Parameter_file Gold_file Test_file

 
For example to use the sample files:

> evalb -p sample.prm sample.gld sample.tst



[3] OPTIONS

You can specify system parameters in the command line options.
Other options concerning to evaluation metrix should be specified
in parameter file, described later.

        -p param_file  parameter file                        
        -d             debug mode                            
        -e n           number of error to kill (default=10)  
        -h             help                                  



[4] OUTPUT FORMAT FROM THE SCORER

The scorer gives individual scores for each sentence, for
example:

  Sent.                        Matched  Bracket   Cross        Correct Tag
 ID  Len.  Stat. Recal  Prec.  Bracket gold test Bracket Words  Tags Accracy
============================================================================
   1    8    0  100.00 100.00     5      5    5      0      6     5    83.33

At the end of the output the === Summary === section gives statistics 
for all sentences, and for sentences <=40 words in length. The summary
contains the following information:

i)   Number of sentences -- total number of sentences.

ii)  Number of Error/Skip sentences -- should both be 0 if there is no
    problem with the parsed/gold files.

iii) Number of valid sentences = Number of sentences - Number of Error/Skip
    sentences 

iv)  Bracketing recall =     (number of correct constituents)
                         ----------------------------------------
                         (number of constituents in the goldfile)

v)   Bracketing precision = (number of correct constituents)
                         ----------------------------------------
                         (number of constituents in the parsed file)

vi)  Complete match = percentaage of sentences where recall and precision are
    both 100%. 

vii) Average crossing = (number of constituents crossing a goldfile constituen
                         ----------------------------------------------------
                                        (number of sentences)

viii) No crossing = percentage of sentences which have 0 crossing brackets.

ix)   2 or less crossing = percentage of sentences which have <=2 crossing brackets.

x)    Tagging accuracy = percentage of correct POS tags (but see [5].3 for exact
     details of what is counted).



[5] HOW TO CREATE A GOLDFILE FROM THE PENN TREEBANK


The gold and parsed files are in a format similar to this:

(TOP (S (INTJ (RB No)) (, ,) (NP (PRP it)) (VP (VBD was) (RB n't) (NP (NNP Black) (NNP Monday))) (. .)))

To create a gold file from the treebank:

tgrep -wn '/.*/' | tgrep_proc.prl 

will produce a goldfile in the required format.  ("tgrep -wn '/.*/'" prints
parse trees, "tgrep_process.prl" just skips blank lines).

For example, to produce a goldfile for section 23 of the treebank:

tgrep -wn '/.*/' | tail +90895 | tgrep_process.prl | sed 2416q > sec23.gold



[6] THE PARAMETER (.prm) FILE


The .prm file sets options regarding the scoring method. COLLINS.prm gives
the same scoring behaviour as the scorer used in (Collins 97). The options 
chosen were: 

1) LABELED 1

to give labelled precision/recall figures, i.e. a constituent must have the
same span *and* label as a constituent in the goldfile.

2) DELETE_LABEL TOP   

Don't count the "TOP" label (which is always given in the output of tgrep) 
when scoring. 

3) DELETE_LABEL -NONE-  

Remove traces (and all constituents which dominate nothing but traces) when
scoring. For example

.... (VP (VBD reported) (SBAR (-NONE- 0) (S (-NONE- *T*-1)))) (. .)))

would be processed to give

.... (VP (VBD reported)) (. .)))


4)
DELETE_LABEL ,     -- for the purposes of scoring remove punctuation
DELETE_LABEL :
DELETE_LABEL ``
DELETE_LABEL ''
DELETE_LABEL .

5) DELETE_LABEL_FOR_LENGTH -NONE-   -- don't include traces when calculating
                                       the length of a sentence (important
                                       when classifying a sentence as <=40
                                       words or >40 words)

6) EQ_LABEL ADVP PRT

Count ADVP and PRT as being the same label when scoring.




[7] MORE DETAILS ABOUT THE SCORING ALGORITHM


1) The scorer initially processes the files to remove all nodes specified
by DELETE_LABEL in the .prm file. It also recursively removes nodes which
dominate nothing due to all their children being removed. For example, if
-NONE- is specified as a label to be deleted, 

.... (VP (VBD reported) (SBAR (-NONE- 0) (S (-NONE- *T*-1)))) (. .)))

would be processed to give

.... (VP (VBD reported)) (. .)))

2) The scorer also removes all functional tags attached to non-terminals
(functional tags are prefixed with "-" or "=" in the treebank). For example
"NP-SBJ" is processed to give "NP", "NP=2" is changed to "NP".


3) Tagging accuracy counts tags for all words *except* any tags which are
deleted by a DELETE_LABEL specification in the .prm file. (For example, for
COLLINS.prm, punctuation tagged as "," ":" etc. would not be included).

4) When calculating the length of a sentence, all words with POS tags not 
included in the "DELETE_LABEL_FOR_LENGTH" list in the .prm file are
counted. (For COLLINS.prm, only "-NONE-" is specified in this list, so
traces are removed before calculating the length of the sentence).

5) There are some subtleties in scoring when either the goldfile or parsed
file contains multiple constituents for the same span which have the same
non-terminal label. e.g. (NP (NP the man)) If the goldfile contains n 
constituents for the same span, and the parsed file contains m constituents
with that nonterminal, the scorer works as follows:

i) If m>n, then the precision is n/m, recall is 100%

ii) If n>m, then the precision is 100%, recall is m/n.

iii) If n==m, recall and precision are both 100%.
