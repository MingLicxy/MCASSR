# import scipy.io as sio
# file_path = '/home/caoxinyu/Arbitrary-scale/data/IXI/2D_mat/train/T2/002_IXI012-HH-1211-T2_slice_047.mat'
# x = sio.loadmat(file_path)['dcm']

#print("###########################################", x.shape) #(256, 256)
#print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", x[128:150, 128:150]) # MRI切片的黑边像素值一般是0

def check_mat_version(file_path):
    with open(file_path, 'rb') as f:
        # 读取文件前128字节
        header = f.read(128)
        # 判断是否为v7.3 (HDF5 格式)
        if b'HDF5' in header:
            return "v7.3 (HDF5)"
        # 检查MATLAB文件的版本信息（版本字节是第124和125字节）
        version = header[124:126]
        if version == b'\x00\x01':
            return "v6"
        elif version == b'\x00\x02':
            return "v7"
        elif version == b'\x00\x03':
            return "v7.3 (非标准检测)"
        else:
            return "未知版本或不支持的格式"

# 示例用法
file_path = '/home/caoxinyu/Arbitrary-scale/data/IXI/2D_mat/train/T2/002_IXI012-HH-1211-T2_slice_047.mat'
version = check_mat_version(file_path)
print(f"{file_path} 的MAT文件版本是: {version}")
