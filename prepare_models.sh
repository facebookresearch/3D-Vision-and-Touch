wget https://dl.fbaipublicfiles.com/ABC/pretrained.targ.gz

mkdir touch_charts/experiments/checkpoint/pretrained/
cp pretrained/touch/encoder_touch touch_charts/experiments/checkpoint/pretrained/encoder_touch

mkdir vision_charts/experiments/checkpoint/pretrained/
cp -r pretrained/vision/ vision_charts/experiments/checkpoint/pretrained

rm -rf pretrained
rm pretrained.targ.gz