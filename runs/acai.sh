# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env bash

acai.py --dataset=lines32 --latent_width=2 --depth=16 --latent=16 --train_dir=TRAIN

acai.py --dataset=mnist32 --latent_width=4 --depth=16 --latent=2 --train_dir=TRAIN
acai.py --dataset=mnist32 --latent_width=4 --depth=16 --latent=16 --train_dir=TRAIN

acai.py --dataset=svhn32 --latent_width=4 --depth=64 --latent=2 --train_dir=TRAIN
acai.py --dataset=svhn32 --latent_width=4 --depth=64 --latent=16 --train_dir=TRAIN

acai.py --dataset=celeba32 --latent_width=4 --depth=64 --latent=2 --train_dir=TRAIN
acai.py --dataset=celeba32 --latent_width=4 --depth=64 --latent=16 --train_dir=TRAIN

acai.py --dataset=cifar10 --latent_width=4 --depth=64 --latent=16 --train_dir=TRAIN
acai.py --dataset=cifar10 --latent_width=4 --depth=64 --latent=64 --train_dir=TRAIN


acai.py --dataset=omniglot32 --latent_width=4 --depth=16 --latent=16 --train_dir=./log



acai.py --dataset=miniimagenet64 --latent_width=4 --depth=64 --latent=64 --train_dir=./log --batch 64 # didn't work, ae loss diverged
acai.py --dataset=miniimagenet32 --latent_width=4 --depth=64 --latent=64 --train_dir=./log
acai.py --dataset=miniimagenet64 --latent_width=4 --depth=64 --latent=16 --train_dir=./log
acai.py --dataset=miniimagenet64 --latent_width=4 --depth=64 --latent=16 --train_dir=./log --advweight=0.1
acai.py --dataset=miniimagenet64 --latent_width=4 --depth=64 --latent=32 --train_dir=./log

acai.py --dataset=vizdoom --latent_width=4 --depth=64 --latent=16 --train_dir=./log # objects weren't reconstructed
acai.py --dataset=vizdoom --latent_width=4 --depth=64 --latent=64 --train_dir=./log

acai.py --dataset=celeba64 --latent_width=4 --depth=64 --latent=16 --train_dir=./log
acai.py --dataset=celeba64 --latent_width=4 --depth=64 --latent=8 --train_dir=./log

acai.py --dataset=miniimagenetgray64 --latent_width=4 --depth=64 --latent=16 --train_dir=./log --advweight=0.1
acai.py --dataset=miniimagenetgray64 --latent_width=4 --depth=64 --latent=16 --train_dir=./log