# in-scale
echo '#######################[div2k-x2]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/others/test_ope.py --exp_folder /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_edsr-baseline-ope_exp_01 --test_config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-ope/test_CIR-SR-div2k-x2.yaml --gpu $1 &&
echo '#######################[div2k-x3]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/others/test_ope.py --exp_folder /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_edsr-baseline-ope_exp_01 --test_config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-ope/test_CIR-SR-div2k-x3.yaml --gpu $1 &&
echo '#######################div2k-x4#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/others/test_ope.py --exp_folder /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_edsr-baseline-ope_exp_01 --test_config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-ope/test_CIR-SR-div2k-x4.yaml --gpu $1 &&
# out-of-scale
echo '#######################[div2k-x6*]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/others/test_ope.py --exp_folder /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_edsr-baseline-ope_exp_01 --test_config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-ope/test_CIR-SR-div2k-x6.yaml --gpu $1 &&
echo '#######################[div2k-x12*]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/others/test_ope.py --exp_folder /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_edsr-baseline-ope_exp_01 --test_config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-ope/test_CIR-SR-div2k-x12.yaml --gpu $1 &&
echo '#######################[div2k-x18*]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/others/test_ope.py --exp_folder /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_edsr-baseline-ope_exp_01 --test_config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-ope/test_CIR-SR-div2k-x18.yaml --gpu $1 &&
echo '#######################[div2k-x24*]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/others/test_ope.py --exp_folder /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_edsr-baseline-ope_exp_01 --test_config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-ope/test_CIR-SR-div2k-x24.yaml --gpu $1 &&
echo '#######################[div2k-x30*]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/others/test_ope.py --exp_folder /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_edsr-baseline-ope_exp_01 --test_config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-ope/test_CIR-SR-div2k-x30.yaml --gpu $1 &&

true
