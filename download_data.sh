#Copyright (c) Facebook, Inc. and its affiliates.
#All rights reserved.
#This source code is licensed under the license found in the
#LICENSE file in the root directory of this source tree. 

wget https://dl.fbaipublicfiles.com/ABC/data.tar.gz
tar xf data.tar.gz -C data --strip-components=1
rm -rf data.tar.gz
