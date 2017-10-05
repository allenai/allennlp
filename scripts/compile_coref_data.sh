#!/bin/bash

function download_and_extract() {
  wget $1/$2
  tar -xvzf $2
  rm $2
}

function compile_partition() {
    rm -f $2.$5.$3$4
    cat conll-2012/$3/data/$1/data/$5/annotations/*/*/*/*.$3$4 >> $2.$5.$3$4
}

function compile_language() {
    compile_partition development dev v4 _gold_conll $1
    compile_partition train train v4 _gold_conll $1
    compile_partition test test v4 _gold_conll $1
}

conll_url=http://conll.cemantix.org/2012/download
download_and_extract $conll_url conll-2012-train.v4.tar.gz
download_and_extract $conll_url conll-2012-development.v4.tar.gz
download_and_extract $conll_url/test conll-2012-test-key.tar.gz
download_and_extract $conll_url/test conll-2012-test-official.v9.tar.gz

download_and_extract $conll_url conll-2012-scripts.v3.tar.gz

download_and_extract http://conll.cemantix.org/download reference-coreference-scorers.v8.01.tar.gz
mv reference-coreference-scorers conll-2012/scorer

ontonotes_path=/projects/WebWare6/ontonotes-release-5.0
bash conll-2012/v3/scripts/skeleton2conll.sh -D $ontonotes_path/data/files/data conll-2012

compile_language english
