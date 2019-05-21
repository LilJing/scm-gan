#!/bin/bash

parse_results() {
    FILENAME="/mnt/nfs/experiments/default/$1/evaluation_metrics*txt"
    #cat $FILENAME | word -1 | count
    # Remove outliers due to wacky Variant A bug
    cat $FILENAME | word -1 | grep -v ^- | count
}

echo "MPC Agent average score"

echo
echo "Vanilla, Training Env"
parse_results scm-gan_05f2fd67

echo
echo "Vanilla, Env Variant A"
parse_results scm-gan_fca09066

echo
echo "TD, Training Env"
parse_results scm-gan_38d67001

echo
echo "Latent Overshooting, Training Env"
parse_results scm-gan_bc539270

echo
echo "TD, Env Variant A"
parse_results scm-gan_79aef573

echo
echo "Latent Overshooting, Env Variant A"
parse_results scm-gan_12224df7

