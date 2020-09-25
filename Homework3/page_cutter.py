import os
from PIL import Image

"""
本想用切割矩阵 + 单行/列识别的方式识别矩阵
但是MathpixAPI太强大了，用不着切割就能识别矩阵
所以此文件废弃
"""


def splitimage(src, rownum, colnum, dstpath):
    img = Image.open(src)
    w, h = img.size
    if rownum <= h and colnum <= w:
        print('Original image info: %sx%s, %s, %s' % (w, h, img.format, img.mode))
        print('开始处理图片切割, 请稍候...')

        s = os.path.split(src)
        if dstpath == '':
            dstpath = s[0]
        fn = s[1].split('.')
        basename = fn[0]
        ext = fn[-1]

        num = 0

        h *= 0.8
        row_height = h // rownum
        col_width = w // colnum
        for r in range(rownum):
            for c in range(colnum):
                box = (c * col_width, r * row_height, (c + 1) * col_width, (r + 1) * row_height)
                img.crop(box).save(os.path.join(dstpath, basename + '_' + str(num) + '.' + ext), ext)
                num = num + 1

        print('图片切割完毕，共生成 %s 张小图片。' % num)
    else:
        print('不合法的行列切割参数！')


src = 'res/matrix.png'
if os.path.isfile(src):
    dstpath = 'out'
    if (dstpath == '') or os.path.exists(dstpath):
        row = int(input('请输入切割行数：'))
        col = int(input('请输入切割列数：'))
        if row > 0 and col > 0:
            splitimage(src, row, col, dstpath)
        else:
            print('无效的行列切割参数！')
    else:
        print('图片输出目录 %s 不存在！' % dstpath)
else:
    print('图片文件 %s 不存在！' % src)
