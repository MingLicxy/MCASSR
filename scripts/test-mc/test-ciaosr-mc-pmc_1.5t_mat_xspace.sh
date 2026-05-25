# in-scale
echo '#######################[div2k-x2]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/test_ciaosr_mc_mat.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-mc/test-pmc_1.5t_mat_xspace/test-ixi-x2.yaml --model $1 --gpu $2 &&
echo '#######################[div2k-x3]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/test_ciaosr_mc_mat.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-mc/test-pmc_1.5t_mat_xspace/test-ixi-x3.yaml --model $1 --gpu $2 &&
echo '#######################[div2k-x4]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/test_ciaosr_mc_mat.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-mc/test-pmc_1.5t_mat_xspace/test-ixi-x4.yaml --model $1 --gpu $2 &&

# out-of-scale
echo '#######################[div2k-x6*]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/test_ciaosr_mc_mat.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-mc/test-pmc_1.5t_mat_xspace/test-ixi-x6.yaml --model $1 --gpu $2 &&
echo '#######################[div2k-x8*]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/test_ciaosr_mc_mat.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-mc/test-pmc_1.5t_mat_xspace/test-ixi-x8.yaml --model $1 --gpu $2 &&
echo '#######################[div2k-x12*]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/test_ciaosr_mc_mat.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-mc/test-pmc_1.5t_mat_xspace/test-ixi-x12.yaml --model $1 --gpu $2 &&
#echo '#######################[div2k-x18*]#######################' &&
#python /home/caoxinyu/Arbitrary-scale/liif-main/test_mc.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-results-A/test-div2k-18.yaml --model $1 --gpu $2 &&
#echo '#######################[div2k-x24*]#######################' &&
#python /home/caoxinyu/Arbitrary-scale/liif-main/test_mc.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-results-A/test-div2k-24.yaml --model $1 --gpu $2 &&
#echo '#######################[div2k-x30*]#######################' &&
#python /home/caoxinyu/Arbitrary-scale/liif-main/test_mc.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-results-A/test-div2k-30.yaml --model $1 --gpu $2 &&

true
