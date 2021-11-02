#!/bin/bash

if [ "$1" = "train" ]; then
    python3 run.py train --train-src=./data/train.de-en.de.wmixerprep --train-tgt=./data/train.de-en.en.wmixerprep --dev-src=./data/valid.de-en.de --dev-tgt=./data/valid.de-en.en --vocab=./data/vocab.json --cuda
elif [ "$1" = "vocab" ]; then
	python3 vocab.py --train-src=./data/train.de-en.de.wmixerprep --train-tgt=./data/train.de-en.en.wmixerprep ./data/vocab.json
else
	echo "Invalid Option Selected"
fi
