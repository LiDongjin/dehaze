import cv2 as cv
import numpy as np
import math
import scipy.sparse
import scipy.sparse.linalg
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt

# 计算环境光的强度
def get_atmo(img, dark_img, percent=0.001):
    [h, w] = img.shape[:2]
    img_size = h*w
    num_pixel = (int)(math.floor(img_size*percent))
    dark_vec = dark_img.reshape(img_size, 1).squeeze()
    img_vec = img.reshape(img_size, 3)
    # 排序得到的下标
    indices = np.argsort(dark_vec)
    # 最大的坐标
    indices = indices[img_size-num_pixel:]

    max_pixel = np.zeros([1, 3])
    for i in range(1, num_pixel):
        if img_vec[indices[i]].sum() > max_pixel.sum():
            max_pixel = img_vec[indices[i]]
    return max_pixel

# def get_atmo(img, dark_img, percent=0.001):
#     # top_pixel_num = (int)(dark_img.shape[0] * dark_img.shape[1] * percent)
#     temp_img = np.squeeze(dark_img.reshape(1, -1))
#     # 倒序排列,得到dark_img中最亮的像素
#     sorted_img = temp_img[np.argsort(temp_img)]
#     # 选出最大值
#     top_pixel = sorted_img[-1]
#     max_axis = np.where(top_pixel == dark_img)
#     # 亮度最大的坐标
#     x = max_axis[0][0]
#     y = max_axis[1][0]
#     # 坐标在输入图像的亮度值
#     return img[x][y]


def get_trans(img, A, w=0.95):
    x = img / A
    t = 1 - w * dark_channel(x, 15)
    return t


def dark_channel(img, size=15):
    r, g, b = cv.split(img)
    min_img = cv.min(r, cv.min(g, b))
    radius = (int)((size - 1) / 2)
    h, w = min_img.shape[:2]
    dc_img = np.zeros((h, w))
    # 处理边界
    temp_img = np.ones((h+radius*2, w+radius*2), dtype='float64')
    for i in range(radius, h+radius):
        for j in range(radius, w+radius):
            temp_img[i][j] = min_img[i-radius][j-radius]
    # 求每个窗口内的最小值
    for i in range(radius, h+radius):
        for j in range(radius, w+radius):
            dc_img[i-radius][j-radius] = np.min(temp_img[i-radius:i+radius, j-radius:j+radius])
    return dc_img


def dehaze(img, t, A, tx=0.1):
    out = np.zeros(img.shape, img.dtype)
    t = cv.max(t, tx)
    for i in range(0, 3):
        out[:, :, i] = (img[:, :, i] - A[i]) / t + A[i]
    return out

def _rolling_block(A, block=(3, 3)):
    """Applies sliding window to given matrix."""
    # 没有padding
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    # A.stride[0] ： 相差行的字节， A.strides[1]：相距列的距离
    strides = (A.strides[0], A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides)
    # 将矩阵分块


def SoftMatting(img, t, eps=10 ** (-7), win_rad=1, lamda=0.0001):
    # 窗口面积
    win_size = (win_rad * 2 + 1) ** 2
    h, w, d = img.shape
    print(img.shape)
    # 窗口边长
    win_diam = win_rad * 2 + 1
    # 每个小窗口的编号
    indsM = np.arange(h * w).reshape((h, w))
    ravelImg = img.reshape(h * w, d)
    win_inds = _rolling_block(indsM, block=(win_diam, win_diam))  # get the convoluted elements index
    win_inds = win_inds.reshape(-1, win_size)
    # 输出 win_inds: 总像素个数 x win_size

    # winI: 总像素个数 x win_size x 3
    winI = ravelImg[win_inds]

    # 计算每个窗口内的均值， 输出总像素个数 x 1 x 3
    win_mu = np.mean(winI, axis=1, keepdims=True)

    # 计算协方差：cov = E(XY)- E(X)E(Y) 输出 win_var: 所有像素点 x 3 x 3
    win_var = np.einsum('...ji,...jk ->...ik', winI, winI) / win_size - np.einsum('...ji,...jk ->...ik', win_mu, win_mu)
    inv = np.linalg.inv(win_var + (eps / win_size) * np.eye(3))

    X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv)
    # ...ij,...kj->...ik 相当于乘 winI - win_mu的转置
    # vals : 所有像素点 x win_size x win_size
    vals = np.eye(win_size) - (1.0 / win_size) * (1 + np.einsum('...ij,...kj->...ik', X, winI - win_mu))

    # 按照win_inds索引将数据转换为矩阵
    # tile: 复制拼接矩阵， ravel: 展开矩阵为一行，
    print("win_id", win_inds.shape)
    nz_indsCol = np.tile(win_inds, win_size).ravel()
    print("nz_inds", nz_indsCol.shape)
    # repeat: 对矩阵中的元素进行连续重复复制，
    nz_indsRow = np.repeat(win_inds, win_size).ravel()
    print("row", nz_indsRow.shape)
    nz_indsVal = vals.ravel()
    L = scipy.sparse.coo_matrix((nz_indsVal, (nz_indsRow, nz_indsCol)), shape=(h * w, h * w))

    m, n = t.shape
    lamda = np.ones([m, n]) * lamda
    confidence = scipy.sparse.diags(lamda.ravel())
    b = lamda.ravel() * t.ravel()
    result = scipy.sparse.linalg.spsolve(L+confidence, b)
    result = np.minimum(np.maximum(result.reshape(m, n), 0), 1)
    return result


if __name__ == "__main__":
    path = "D:/Project/pycharm/ImageProcessing/trees.png"
    output = "D:/Project/pycharm/ImageProcessing/output/softmatting.jpg"

    img = cv.imread(path)
    img = img.astype('float64') / 255
    dark_img = dark_channel(img)
    A = get_atmo(img, dark_img)
    t = get_trans(img, A)
    new_t = SoftMatting(img, t)
    # 去霧
    out1 = dehaze(img, t, A, 0.1)
    out2 = dehaze(img, new_t, A, 0.1)
    test1 = np.hstack((dark_img, t, new_t))
    cv.imshow("t<->soft_matting", test1)
    cv.imwrite(output, test1*255)
    out = np.hstack((img, out1, out2))
    cv.imwrite("./output/dehaze.jpg", out*255)
    cv.waitKey()

