#!/usr/bin/env bash

infer.py --ae_dir ./log/omniglot32/ACAI_advdepth16_advweight0.5_depth16_latent16_reg0.2_scales3 --dataset omniglot32
infer.py --ae_dir ./log/miniimagenet32/ACAI_advdepth64_advweight0.5_depth64_latent64_reg0.2_scales3 --dataset miniimagenet32