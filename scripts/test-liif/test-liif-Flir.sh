# in-scale
# echo '#######################[div2k-x2]#######################' &&
# python /home/caoxinyu/Arbitrary-scale/liif-main/test.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-Flir/test-div2k-2.yaml --model $1 --gpu $2 &&
# echo '#######################[div2k-x3]#######################' &&
# python /home/caoxinyu/Arbitrary-scale/liif-main/test.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-Flir/test-div2k-3.yaml --model $1 --gpu $2 &&
# echo '#######################[div2k-x4]#######################' &&
# python /home/caoxinyu/Arbitrary-scale/liif-main/test.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-Flir/test-div2k-4.yaml --model $1 --gpu $2 &&

# # out-of-scale
echo '#######################[div2k-x6*]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/test.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-Flir/test-div2k-6.yaml --model $1 --gpu $2 &&
echo '#######################[div2k-x8*]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/test.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-Flir/test-div2k-8.yaml --model $1 --gpu $2 &&
echo '#######################[div2k-x12*]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/test.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-Flir/test-div2k-12.yaml --model $1 --gpu $2 &&
# echo '#######################[div2k-x18*]#######################' &&
# python /home/caoxinyu/Arbitrary-scale/liif-main/test.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-Flir/test-div2k-18.yaml --model $1 --gpu $2 &&
# echo '#######################[div2k-x24*]#######################' &&
# python /home/caoxinyu/Arbitrary-scale/liif-main/test.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-Flir/test-div2k-24.yaml --model $1 --gpu $2 &&
# echo '#######################[div2k-x30*]#######################' &&
# python /home/caoxinyu/Arbitrary-scale/liif-main/test.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-Flir/test-div2k-30.yaml --model $1 --gpu $2 &&

# 非整数尺度
# echo '#######################[div2k-x1.7]#######################' &&
# python /home/caoxinyu/Arbitrary-scale/liif-main/test.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-Flir/test-div2k-1.7.yaml --model $1 --gpu $2 &&
# echo '#######################[div2k-x3.1]#######################' &&
# python /home/caoxinyu/Arbitrary-scale/liif-main/test.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-Flir/test-div2k-3.1.yaml --model $1 --gpu $2 &&
# echo '#######################[div2k-x4.5]#######################' &&
# python /home/caoxinyu/Arbitrary-scale/liif-main/test.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-Flir/test-div2k-4.5.yaml --model $1 --gpu $2 &&
# echo '#######################[div2k-x5.9]#######################' &&
# python /home/caoxinyu/Arbitrary-scale/liif-main/test.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-Flir/test-div2k-5.9.yaml --model $1 --gpu $2 &&
# echo '#######################[div2k-x7.3]#######################' &&
# python /home/caoxinyu/Arbitrary-scale/liif-main/test.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-Flir/test-div2k-7.3.yaml --model $1 --gpu $2 &&
true
