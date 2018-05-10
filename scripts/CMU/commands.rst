# to setup the system
git clone https://github.com/prasoontelang/allennlp.git

# download the dataset from the website
STORIES="stories"
TAR="${STORIES}.tar.gz"
wget -O ${TAR} https://conversationalmlcourse.bitbucket.io/datasets/stories_dataset.tar.gz
mkdir $STORIES
tar -C $STORIES -zxvf $TAR

# to train your model
allennlp train training_config/bidaf.json --serialization-dir /tmp/tutorials/bidaf

# to evaluate your model
allennlp evaluate --evaluation-data-file ~/dataset/small_dev.json /tmp/tutorials/bidaf_small/model.tar.gz
