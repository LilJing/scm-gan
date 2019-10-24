#!/bin/bash

#gnomehat -m "Eval SC2 110k Ablation Vanilla" python main.py --load-from /mnt/nfs/experiments/default/scm-gan_7b19101d --env sc2_star_intruders --evaluate --evaluations=10
#gnomehat -m "Eval SC2 110k CF_Shuffle Vanilla" python main.py --load-from /mnt/nfs/experiments/default/scm-gan_c31c67dd --env sc2_star_intruders --evaluate --evaluations=10
#gnomehat -m "Eval SC2 110k CF_Sparsity Vanilla" python main.py --load-from /mnt/nfs/experiments/default/scm-gan_b2234301 --env sc2_star_intruders --evaluate --evaluations=10
#gnomehat -m "Eval SC2 110k CF_ControlBias Vanilla" python main.py --load-from /mnt/nfs/experiments/default/scm-gan_afa2a682 --env sc2_star_intruders --evaluate --evaluations=10

echo "ABLATION 110k"
cat /mnt/nfs/experiments/default/scm-gan_dd6a6dae/eval*txt | word -1 | count
EXPERIMENT=scm-gan_dd6a6dae
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

echo "ABLATION 100k"
EXPERIMENT=scm-gan_e78703cc
cat /mnt/nfs/experiments/default/$EXPERIMENT/eval*txt | word -1 | count
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


echo "CF SHUFFLE"
cat /mnt/nfs/experiments/default/scm-gan_9788da55/eval*txt | word -1 | count
EXPERIMENT=scm-gan_9788da55
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


echo "CF SPARSITY"
cat /mnt/nfs/experiments/default/scm-gan_6cd3c518/eval*txt | word -1 | count
EXPERIMENT=scm-gan_6cd3c518
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


echo "CF CONTROL-BIAS"
cat /mnt/nfs/experiments/default/scm-gan_b5177a12/eval*txt | word -1 | count
EXPERIMENT=scm-gan_b5177a12
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



echo "Ablation 110k VariantA"
EXPERIMENT="scm-gan_c38734a4"
cat /mnt/nfs/experiments/default/$EXPERIMENT/eval*txt | word -1 | count
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



echo "CF SHUFFLE VariantA"
EXPERIMENT=scm-gan_0ea009c0
cat /mnt/nfs/experiments/default/$EXPERIMENT/eval*txt | word -1 | count
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

echo "CF SPARSITY VariantA"
EXPERIMANT=scm-gan_98a7089d
cat /mnt/nfs/experiments/default/$EXPERIMENT/eval*txt | word -1 | count
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


