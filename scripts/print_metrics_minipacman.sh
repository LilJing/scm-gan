#!/bin/bash

echo "Printing metrics for MiniPacman"



echo "MiniPacMan Ablation:"
EXPERIMENT=scm-gan_5e3afb0f
echo "MSE H=3: "
cat /mnt/nfs/experiments/default/$EXPERIMENT/*json | head -4 | tail -1
echo "MSE H=5"
cat /mnt/nfs/experiments/default/$EXPERIMENT/*json | head -6 | tail -1
echo "MSE H=10"
cat /mnt/nfs/experiments/default/$EXPERIMENT/*json | head -11 | tail -1
echo "MSE H=20"
cat /mnt/nfs/experiments/default/$EXPERIMENT/*json | head -21 | tail -1
echo "Score"
cat /mnt/nfs/experiments/default/$EXPERIMENT/eval*.txt | word -1 | count
echo
echo

echo "MiniPacMan Action-Reward:"
EXPERIMENT=scm-gan_893fd86a
echo "MSE H=3: "
cat /mnt/nfs/experiments/default/$EXPERIMENT/*json | head -4 | tail -1
echo "MSE H=5"
cat /mnt/nfs/experiments/default/$EXPERIMENT/*json | head -6 | tail -1
echo "MSE H=10"
cat /mnt/nfs/experiments/default/$EXPERIMENT/*json | head -11 | tail -1
echo "MSE H=20"
cat /mnt/nfs/experiments/default/$EXPERIMENT/*json | head -21 | tail -1
echo "Score"
cat /mnt/nfs/experiments/default/$EXPERIMENT/eval*.txt | word -1 | count
echo
echo
echo "MiniPacMan Disentanglement:"
EXPERIMENT=scm-gan_f62f571e
echo "MSE H=3: "
cat /mnt/nfs/experiments/default/$EXPERIMENT/*json | head -4 | tail -1
echo "MSE H=5"
cat /mnt/nfs/experiments/default/$EXPERIMENT/*json | head -6 | tail -1
echo "MSE H=10"
cat /mnt/nfs/experiments/default/$EXPERIMENT/*json | head -11 | tail -1
echo "MSE H=20"
cat /mnt/nfs/experiments/default/$EXPERIMENT/*json | head -21 | tail -1
echo "Score"
cat /mnt/nfs/experiments/default/$EXPERIMENT/eval*.txt | word -1 | count
echo
echo
echo "MiniPacMan Sparsity:"
EXPERIMENT=scm-gan_09a41c1c
echo "MSE H=3: "
cat /mnt/nfs/experiments/default/$EXPERIMENT/*json | head -4 | tail -1
echo "MSE H=5"
cat /mnt/nfs/experiments/default/$EXPERIMENT/*json | head -6 | tail -1
echo "MSE H=10"
cat /mnt/nfs/experiments/default/$EXPERIMENT/*json | head -11 | tail -1
echo "MSE H=20"
cat /mnt/nfs/experiments/default/$EXPERIMENT/*json | head -21 | tail -1
echo "Score"
cat /mnt/nfs/experiments/default/$EXPERIMENT/eval*.txt | word -1 | count
echo
echo
