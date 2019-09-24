#!/bin/bash

echo "Printing metrics for Pong"



echo "Pong Ablation:"
EXPERIMENT=
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

echo "Pong Action-Reward:"
EXPERIMENT=
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
echo "Pong Disentanglement:"
EXPERIMENT=
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
echo "Pong Sparsity:"
EXPERIMENT=
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
