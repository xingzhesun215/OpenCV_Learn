# opencv-python 4.5.5.64
# numpy 1.24.2
# Pycharm 2021.3.2(Community Edition)
import cv2
import cv2 as cv
import numpy as np

# 20230214
print("Hello OpenCV")

a = 10

# 10-图像缩放resize
# dst=cv2.resize(src,dsize) dsize=(size_y,size_x)
# dst=cv2.resize(src,dsize,fx,fy)  fx,fy缩放大小
if a == 10:
    lena = cv.imread("lena.jpg")
    cv.imshow("lena", lena)
    # 指定大小缩放
    size = (300, 300)
    size_200 = cv2.resize(lena, size)
    cv.imshow("resize", size_200)
    # 按比例缩放
    rows, cols = lena.shape[:2]
    resize1 = cv2.resize(lena, (round(cols * 0.5), round(rows * 0.5)))
    cv.imshow("resize1 0.5 0.5", resize1)

    # 按比例缩放2 设置行，后设置列，行为0.5倍，列为0.3倍：
    resize2 = cv2.resize(lena, None, fx=0.5, fy=0.3)
    cv.imshow("resize2 0.5 0.3", resize2)

    cv.waitKey()
    cv.destroyAllWindows()

# 9-图片转换
# BGR转灰度图  BGR转为RGB RGB转回BGR 甚至其他色彩空间也可以转换

if a == 9:
    lena = cv.imread("lena.jpg")  # 此时得到的bgr数据
    cv.imshow("lena bgr", lena)
    lena_gray = cv.cvtColor(lena, cv2.COLOR_BGR2GRAY)  # 从BGR转为灰度图
    print(lena_gray.shape)
    print(lena_gray.dtype)
    print(lena_gray)
    print("----------------")
    lena_gray_bgr = cv.cvtColor(lena_gray, cv2.COLOR_GRAY2BGR)
    print(lena_gray_bgr.shape)  # 通道变回3了
    print(lena_gray_bgr.dtype)
    print(lena_gray_bgr)  # 但数据变不回原来的了
    print("----------------")

    lena_rgb = cv.cvtColor(lena, cv2.COLOR_BGR2RGB)  # BGR转到RGB
    cv.imshow("lena rgb", lena_rgb)
    lena_bgr = cv.cvtColor(lena_rgb, cv2.COLOR_RGB2BGR)  # RGB转到BGR
    cv.imshow("lena bgr2", lena_bgr)  # 和lena原图一毛一样
    cv.waitKey()
    cv.destroyAllWindows()

# 8-图片融合
# 结果图像=图像1*系数1+图片2*系数2+亮度调节
# 亮度调节就是每个点的各个通道额外加的数
if a == 8:
    lena = cv.imread("lena.jpg")
    cv.imshow("lena", lena)
    print(lena)
    pic_512 = cv.imread("512_512.png")
    cv.imshow("512_512", pic_512)
    result = cv.addWeighted(lena, 1, pic_512, 0, 0)  # 最后一位为亮度
    print("-----------------------------")
    print(result)
    cv.imshow("result", result)
    cv.waitKey()
    cv.destroyAllWindows()

# 7-图像加法运算 numpy加法(取模算法) opencv加法(饱和算法)
if a == 7:
    lena = cv.imread("lena.jpg")
    np_add = lena + lena  # 每个坐标点的各通道相加结果和256取模
    print(lena)
    print("-------------------")
    print(np_add)
    print("-------------------")
    cv_add = cv.add(lena, lena)  # 每个坐标点的各通道相加结果大于255取255
    print(lena)
    print("-------------------")
    print(cv_add)
    print("-------------------")
    cv.imshow("np_add", np_add)
    cv.imshow("cv_add", cv_add)
    cv.waitKey()
    cv.destroyAllWindows()

# 6-使用numpy生成空白通道,并进行合并操作
if a == 6:
    moon = cv.imread("moon.png")
    # 分别取得行 列 通道数
    rows, cols, chn = moon.shape
    b = cv.split(moon)[0]
    cv.imshow("moon_b", b)
    # 空白g r通道数据
    g = np.zeros((rows, cols), moon.dtype)
    r = np.zeros((rows, cols), moon.dtype)
    m = cv.merge([b, g, r])
    cv.imshow("merge", m)
    cv.waitKey()
    cv.destroyAllWindows()

# 5-通道的拆分和合并
# 拆分split:将3个通道独立出来
# 合并merge:将3个通道合并回彩色通道
if a == 5:
    hzw = cv.imread("hzw.png")
    # 分别得到b g r通道的数据,如果需要得到单个通道数据,加上下标即可:cv.split(xx)[index]
    hzw_b, hzw_g, hzw_r = cv.split(hzw)
    cv.imshow("hzw原始图片", hzw)
    cv.imshow("hzw_b", hzw_b)
    cv.imshow("hzw_g", hzw_g)
    cv.imshow("hzw_r", hzw_r)

    # 以b,g,r的通道组合合并
    hzw_bgr = cv.merge([hzw_b, hzw_g, hzw_r])
    # 以r,g,b的通道组合合并
    hzw_rgb = cv.merge([hzw_r, hzw_g, hzw_b])
    cv.imshow("hzw_bgr", hzw_bgr)
    cv.imshow("hzw_rgb", hzw_rgb)
    cv.waitKey()
    cv.destroyAllWindows()

# 4-图像ROI(region of interest)-感兴趣区域
# 把月球视觉的地图扣到露娜头上
if a == 4:
    moon = cv.imread("moon.png")
    # diqiu为100行100列3个通道的空白区域
    diqiu = np.ones((100, 100, 3))
    lena = cv.imread("lena.jpg")
    # 取moon图片的地球区域 大概是y:0-100 x:100-200
    diqiu = moon[0:100, 100:200]
    # 将b显示在a的同样大小左上角部分
    lena[0:100, 50:150] = diqiu
    cv.imshow("moon", moon)
    cv.imshow("lena", lena)
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
