## Script for downloading data

# VCR
mkdir annotations
wget https://s3.us-west-2.amazonaws.com/ai2-rowanz/vcr1annots.zip
unzip vcr1annots.zip -d annotations
rm vcr1annots.zip