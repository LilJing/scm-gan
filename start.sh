#!/bin/bash

echo 'Running Pre-trained Model-Based agent on a StarCraft II environment'


echo 'Checking Python requirements...'
pip install -r  requirements.txt

echo 'Checking for ffmpeg...'
which ffmpeg || (echo "WARNING: ffmpeg not found. Run apt install ffmpeg" && exit)

echo 'Checking StarCraftII installation...'
(ls ~/StarCraftII/Versions/ | grep Base | sort | tail -1 && echo "StarCraft is installed") || (echo "Warning: could not find StarCraftII - run install_starcraft")

echo 'Downloading pretrained model...'
pushd pretrained_models/sc2_star_intruders
wget -nc https://lwneal.com/sc2_star_intruders.tar.gz
tar xzvf sc2_star_intruders.tar.gz
popd

echo 'Starting evaluation...'
python main.py --evaluate --env sc2_star_intruders --load-from pretrained_models/sc2_star_intruders
