#!/usr/bin/env bash

if [ ! -f models/glove.6B.zip ]; then
    wget -P models http://nlp.stanford.edu/data/glove.6B.zip
fi
unzip models/glove.6B.zip -d models
rm models/glove.6B.zip