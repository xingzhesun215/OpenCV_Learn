# opencv-python 4.5.5.64
# numpy 1.24.2
# Pycharm 2021.3.2(Community Edition)
import cv2 as cv
import numpy as np

# 20230214
print("Hello OpenCV")

a = 4

# 图像ROI(region of interest)-感兴趣区域
if a == 4:
    a = cv.imread("lena.jpg")
    # b为101行101列3个通道的区域
    b = np.ones((101, 101, 3))
    yueqiu = cv.imread("xxx.jpg")
    # 取b是a中100-200行 130到200列的部分
    b = a[100:200, 130:200]
    # 将b显示在a的同样大小左上角部分
    a[0:100, 0:70] = b
    yueqiu[0:100, 0:70] = b
    cv.imshow("original", a)
    cv.imshow("yueqiu", yueqiu)
    cv.waitKey()
    cv.destroyAllWindows()

# 3-修改图片的点元素或者区块通道修改
if a == 3:
    face = cv.imread("lena.jpg")
    p = face[100, 200]  # opencv坐标[y,x]
    print(p)  # [118 119 209]
    face[100, 200] = [0, 1, 2]  # 变成一个近似黑点
    face[110:150, 220:280] = [0, 1, 2]  # 这一块区域变成了近似黑块
    print(face.item(100, 200, 0))  # 用numpy高效一些
    print(face.item(100, 200, 1))
    print(face.item(100, 200, 2))
    face.itemset((100, 200, 0), 3)  # y方向100 x方向200的第一个通道颜色值为3
    p = face[100, 200]
    print(p)  # [3 1 2]

    cv.imshow("lena's face", face)  # 确有黑点
    cv.waitKey(0)
    cv.destroyAllWindows()

# 2-显示读取的图片的属性
if a == 2:
    # 原图 彩图使用三原色表示RGB,在opencv使用BGR进行通道标识
    lena = cv.imread("lena.jpg", -1)
    print(lena.shape)  # (512, 512, 3) 就是形状(行，列，通道数)
    print(lena.size)  # 786432  就是512*512*3
    print(lena.dtype)  # uint8 图像数据类型
    print(lena[0])  # [[b g r] [b g r]....[b g r]]
    print(lena[0][0])  # [128 138 225]
    print(lena)  # [[x x...x x],[x x...x x]...[x x...x x] [x x...x x]]   512*512   x=[b g r]

    print("---------------------------")

    # 灰度图
    lena = cv.imread("lena.jpg", 0)
    print(lena.shape)  # (512, 512) 就是形状(行，列，通道数=1)
    print(lena.size)  # 262144   就是512*512
    print(lena.dtype)  # uint8 图像数据类型
    print(lena[0])  # [xx xx ... xx xx 512]
    print(lena)  # [[],[]...[] []]   512*512

# 1-导入图像 显示图像 保存图像
if a == 1:
    # 读取图片 可不传图片格式参数,默认为原图 -1为原图 0为灰度图
    lena = cv.imread("lena.jpg", -1)
    # 显示图片 窗口名
    cv.imshow("lena's face", lena)
    # 窗口显示时间,单位为ms 0/-1为一直显示
    cv.waitKey(0)
    # 销毁窗口
    cv.destroyAllWindows()
    # 保存图片
    cv.imwrite("1_other_lena.jpg", lena)
