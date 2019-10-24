#!/bin/bash
#gnomehat -m "Evaluate SC2 Ablation" python main.py --env sc2_star_intruders --evaluate --load-from /mnt/nfs/experiments/counterfactual-L1-ICLR-20190920/scm-gan_009c0750 --evaluations=10
#gnomehat -m "Evaluate SC2 CF Sparsity .01" python main.py --env sc2_star_intruders --evaluate --load-from /mnt/nfs/experiments/counterfactual-L1-ICLR-20190920/scm-gan_f6cba63d --evaluations=10

#gnomehat -m "Evaluate SC2 CF Sparsity .01 Variant A" python main.py --env sc2_star_intruders_variant_a --evaluate --load-from /mnt/nfs/experiments/counterfactual-L1-ICLR-20190920/scm-gan_f6cba63d --evaluations=10
#gnomehat -m "Evaluate SC2 CF Sparsity .01 Variant B" python main.py --env sc2_star_intruders_variant_b --evaluate --load-from /mnt/nfs/experiments/counterfactual-L1-ICLR-20190920/scm-gan_f6cba63d --evaluations=10
#gnomehat -m "Evaluate SC2 CF Sparsity .01 Variant C" python main.py --env sc2_star_intruders_variant_c --evaluate --load-from /mnt/nfs/experiments/counterfactual-L1-ICLR-20190920/scm-gan_f6cba63d --evaluations=10

#gnomehat -m "Evaluate SC2 Ablation Variant A" python main.py --env sc2_star_intruders_variant_a --evaluate --load-from /mnt/nfs/experiments/counterfactual-L1-ICLR-20190920/scm-gan_009c0750 --evaluations=10
#gnomehat -m "Evaluate SC2 Ablation Variant B" python main.py --env sc2_star_intruders_variant_b --evaluate --load-from /mnt/nfs/experiments/counterfactual-L1-ICLR-20190920/scm-gan_009c0750 --evaluations=10
#gnomehat -m "Evaluate SC2 Ablation Variant C" python main.py --env sc2_star_intruders_variant_c --evaluate --load-from /mnt/nfs/experiments/counterfactual-L1-ICLR-20190920/scm-gan_009c0750 --evaluations=10




gnomehat -m "Evaluate SC2 Disentanglement=.01" python main.py --env sc2_star_intruders --evaluate --load-from /mnt/nfs/experiments/default/scm-gan_eb5fa12d --evaluations=10
gnomehat -m "Evaluate SC2 Disentanglement=.01 Variant A" python main.py --env sc2_star_intruders_variant_a --evaluate --load-from /mnt/nfs/experiments/default/scm-gan_eb5fa12d --evaluations=10
gnomehat -m "Evaluate SC2 Disentanglement=.01 Variant B" python main.py --env sc2_star_intruders_variant_b --evaluate --load-from /mnt/nfs/experiments/default/scm-gan_eb5fa12d --evaluations=10
gnomehat -m "Evaluate SC2 Disentanglement=.01 Variant C" python main.py --env sc2_star_intruders_variant_c --evaluate --load-from /mnt/nfs/experiments/default/scm-gan_eb5fa12d --evaluations=10
