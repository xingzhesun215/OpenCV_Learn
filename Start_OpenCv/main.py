# opencv-python 4.5.5.64
# numpy 1.24.2
# Pycharm 2021.3.2(Community Edition)
import cv2
import cv2 as cv
import numpy as np

# 20230214
print("Hello OpenCV")

a = 20

if a == 20:
    img = cv.imread("pic_240.png")
    rows, cols = img.shape[:2]
    print(rows, cols)
    pts1 = np.float32([[50, 10], [0, 229], [229, 0], [190, 229]])
    pts2 = np.float32([[0, 0], [229, 0], [0, 229], [229, 229]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    move = cv.warpPerspective(img, M, (cols, rows))
    cv.imshow("Original", img)
    cv.imshow("Perspective", move)
    cv.waitKey()
    cv.destroyAllWindows()

if a == 19:
    img = cv.imread("lena.jpg")
    hei, wid = img.shape[:2]
    p1 = np.float32([[0, 0], [wid - 1, 0], [0, hei - 1]])
    p2 = np.float32([[0, hei * 0.33], [wid * 0.85, hei * 0.25], [wid * 0.15, hei * 0.7]])
    M = cv.getAffineTransform(p1, p2)
    print(M)  # [[ 8.51663429e-01  1.50293548e-01  0.00000000e+00] [-8.01565689e-02  3.70724045e-01  1.68960007e+02]]
    move = cv.warpAffine(img, M, (wid, hei))
    cv.imshow("lena", img)
    cv.imshow("lena rotation", move)
    cv.waitKey()
    cv.destroyAllWindows()

# 18-绘制图形及文字
if a == 18:
    canvas = np.zeros((600, 600, 3), np.uint8)  # 全0的黑色背景
    # canvas = np.ones((600, 600, 3), np.uint8) * 255  # 全白背景

    # 绘制直线
    cv.line(canvas, (0, 0), (200, 90), (255, 0, 0), 5)  # cv2.line(绘制图层,(起点x,起点y),(终点x,终点y),(b,g,r),粗细大小)

    # 绘制矩阵
    cv.rectangle(canvas, (250, 250), (300, 430), (0, 255, 255),
                 2)  # cv2.rectangle(绘制图层,(左上角x,左上角y),(右下角x,右下角y),(b,g,r),划线粗细) 划线粗细为-1时表示实心矩阵

    # 绘制圆形
    cv.circle(canvas, (400, 400), 50, (0, 0, 255), -1)  # cv2.circle(绘制图层,(圆心x,圆心y),半径,(b,g,r),划线粗细) 划线粗细为负数时表示实心圆

    # 绘制椭圆
    cv.ellipse(canvas, (256, 256), (100, 70), 30, 0, 360, 255,
               -1)  # cv2.ellipse(绘制图层,(椭圆圆心x,椭圆圆心y),(长轴长,短轴长),椭圆整体旋转角度,椭圆绘制开始角度,椭圆绘制停止角度,颜色,划线粗细)划线粗细为-1时表示实心椭圆

    # 绘制其他边形
    pts = np.array([[30, 50], [30, 250], [130, 250], [130, 350], [230, 350]], np.int32)
    cv.polylines(canvas, [pts], True, (100, 100, 100), 1)  # cv2.polylines(绘制图层,点集,是否闭合,(b,g,r),划线粗细)

    # 绘制文字 cv.putText(绘制图层,内容,(文字左下角x,文字左下角y),字体样式,字体大小,字体颜色,线条宽度)
    cv.putText(canvas, "HelloOpenCV", (100, 100), cv2.FONT_HERSHEY_TRIPLEX, 2, (66, 66, 66), 3)
    cv.imshow("Canvas", canvas)
    cv.waitKey()
    cv.destroyAllWindows()

# 17-图像平滑-中值滤波 medianBlur
if a == 17:
    nat = cv.imread("lena.jpg")
    r = cv.medianBlur(nat, 9)
    cv2.imshow("original", nat)
    cv2.imshow("result", r)
    cv2.waitKey()
    cv2.destroyAllWindows()

# 16-图像滤波-高斯滤波GaussianBlur
if a == 16:
    nat = cv.imread("lena.jpg")
    r = cv.GaussianBlur(nat, (5, 5), 0)
    cv2.imshow("original", nat)
    cv2.imshow("result", r)
    cv2.waitKey()
    cv2.destroyAllWindows()

# 15-图像平滑-方框滤波boxFilter
if a == 15:
    nat = cv.imread("lena.jpg")
    r = cv.boxFilter(nat, -1, (2, 2), normalize=0)
    cv2.imshow("original", nat)
    cv2.imshow("result", r)
    cv2.waitKey()
    cv2.destroyAllWindows()

# 14-图像平滑-均值滤波(暂时不太理解,后续再细究)
if a == 14:
    nat = cv.imread("lena.jpg")
    r = cv.blur(nat, (5, 5))
    cv2.imshow("original", nat)
    cv2.imshow("result", r)
    cv2.waitKey()
    cv2.destroyAllWindows()

# 13-简单的阈值处理
# 对于每个像素，应用相同的阈值。如果像素的值小于阈值，它就被设置为0，否则就被设置为一个最#大值。
# 函数cv.threshold被用来应用阈值化。第一个参数是源图像，它应该是一个灰度图像。第二个参数是阈值，用于对像素值进行分类。
# 第三个参数是最大值，它被分配给超过阈值的像素值。第四个参数由OpenCV提供了不同类型的阈值处理。
if a == 13:
    color_0_255 = cv.imread("0_255.png", 0)  # 灰度图像素带
    r, b1 = cv2.threshold(color_0_255, 127, 255, cv2.THRESH_BINARY)  # 二进制阈值化:比阈值大设为最大值,否则为0
    r, b2 = cv2.threshold(color_0_255, 127, 255, cv2.THRESH_BINARY_INV)  # 反二进制阈值化:比阈值大设为0,否则为最大值
    r, b3 = cv2.threshold(color_0_255, 127, 255, cv2.THRESH_TRUNC)  # 截断阈值化:比阈值大的都设置成阈值
    r, b4 = cv2.threshold(color_0_255, 127, 255, cv2.THRESH_TOZERO)  # 反阈值化为0:大于阈值的为0
    r, b5 = cv2.threshold(color_0_255, 127, 255, cv2.THRESH_TOZERO_INV)  # 阈值化为0:小于阈值则设为0
    cv.imshow("orginal 0~255", color_0_255)
    cv.imshow("BINARY", b1)
    cv.imshow("BINARY_INV", b2)
    cv.imshow("TRUNC", b3)
    cv.imshow("TOZERO", b4)
    cv.imshow("TOZERO_INV", b5)

    cv.waitKey()
    cv.destroyAllWindows()

# 12-图像翻转
# flipCode = 0 :以x轴上下翻转
# flipCode > 0 :以y轴左右翻转
# flipCode < 0 :先水平 再左右翻转
if a == 12:
    lufei = cv.imread("lufei.png")
    cv.imshow("lufei", lufei)
    lufei_x = cv.flip(lufei, 0)
    lufei_y = cv.flip(lufei, 1)
    lufei_xy = cv.flip(lufei, -1)
    cv.imshow("lufei up_down", lufei_x)
    cv.imshow("lufei left_right", lufei_y)
    cv.imshow("lufei up_down_left_right", lufei_xy)

    cv.waitKey()
    cv.destroyAllWindows()

# 11-图像缩放resize
# dst=cv2.resize(src,dsize) dsize=(size_y,size_x)
# dst=cv2.resize(src,dsize,fx,fy)  fx,fy缩放大小
if a == 11:
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

# 10-色彩空间转换-寻找蓝色的物体
# HSV比BGR更容易表示一种颜色
# 1,读图(或者读视频帧) 2,从BGR转为HSV 3,对HSV图像中的蓝色范围进行阈值处理 4,单独提取蓝色物体,可以在该图像上操作
if a == 10:
    frame = cv.imread("blue_test.png")
    cv.imshow("frame", frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 定义hsv中蓝色的范围
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    # 得到唯一的蓝色图片
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    res = cv.bitwise_and(frame, frame, mask)
    cv.imshow("mask", mask)
    cv.imshow("res", res)

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
