# -*-coding:utf-8-*-
import cv2
import sys
import numpy as np

if sys.version > '3':
    PY3 = True
else:
    PY3 = False

import os
import sys

if sys.version > '3':
    PY3 = True
else:
    PY3 = False


class HeatMap(object):
    def __init__(self,
                 data,
                 image,
                 width=0,
                 height=0,
                 max_color=240,
                 ):
        u"""
        data: list, [[x,y],[x1,y1],[x2,y2]]
        image: Mat, image,  background
        width: int, background width
        height: int, background height
        max_color: int, color hsl round
        """

        # assert type(data) in (list, tuple)
        assert self.is_num(width) and self.is_num(height)
        assert width >= 0 and height >= 0

        count = 0
        data2 = []
        for hit in data:
            if len(hit) == 3:
                x, y, n = hit
            elif len(hit) == 2:
                x, y, n = hit[0], hit[1], 1
            else:
                raise Exception(u"length of hit is invalid!")

            data2.append((x, y, n))
            count += n

        self.data = data2
        self.count = count
        self.image = image
        self.width = width
        self.height = height
        self.max_color = max_color
        self.save_as = None

        if self.image is None and (self.width == 0 or self.height == 0):
            w, h = self.get_max_size(data)
            self.width = self.width or w
            self.height = self.height or h
            self.image = np.zeros((self.height, self.width, 3), "uint8")
            # print(self.image.shape)

    def __mk_img(self, image=None):
        u"""生成临时图片
        image: Mat, image,  background
        """

        image = image or self.image
        if image is not None:
            self.height, self.width = self.image.shape[:2]
        image = np.zeros(image.shape, "uint8")
        self.__im = image

    def __heat(self, heat_data, x, y, n, template):
        u"""
        heat_data: list, [x1,x2,x3,x4], sizeof = self.width*self.height
        x: int,
        y: int,
        n: int, number
        template: dict, circle
        """

        l = len(heat_data)
        width = self.width
        hight = self.height
        # print(template)
        # p = width * y + x
        for ip, (iv,ix,iy) in template:
            px = x + ix
            py = y + iy
            p2 = width * py + px
            if 0 <= p2 < l and 0 < px < width and 0 < py < hight:
                heat_data[p2] += iv*n

    def __paint_heat(self, heat_data, gamma_inv=2.):
        u"""
        heat_data: list, [x1,x2,x3,x4], sizeof = self.width*self.height
        gamma_inv: float, color transform ,more min, image more red
        """

        width = self.width
        height = self.height

        max_v = max(heat_data)
        heat_data = np.reshape(heat_data, (height, width))
        if max_v <= 0:
            # 空图片
            return

        heat_data = heat_data / max_v
        heat_data = np.log(1. + heat_data) / np.log(gamma_inv)
        heat_data = np.clip(heat_data, 0, 1)
        image = np.zeros(heat_data.shape + (3,), np.uint8)
        scale = self.max_color / 360. * 255
        heat_data = np.clip(heat_data * scale, 0, 255).round().astype('uint8')

        image[:, :, 0] = heat_data
        image[:, :, 1] = 127
        image[:, :, 2] = 255
        image[heat_data == 0] = 0
        self.__im = cv2.cvtColor(image, cv2.COLOR_HLS2RGB_FULL)

    def heatmap(self, save_as=None, image=None, data=None, r=10, gamma_inv=2.):
        """
        save_as: save path
        data: point data, [x,y,number] or [x,y]
        r: heat radius
        gamma_inv: float, color transform ,more min, image more red
        """
        self.__mk_img(image)
        circle = self.mk_circle(r, self.width)
        heat_data = np.zeros(self.width * self.height, np.float)
        data = data or self.data
        data_count = self.point_statistic(data)
        mask = np.where(data_count > 0)

        for i in range(len(mask[0])):
            x = int(mask[0][i])
            y = int(mask[1][i])
            n = int(data_count[x, y])
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                continue

            self.__heat(heat_data, x, y, n, circle)

        self.__paint_heat(heat_data, gamma_inv=gamma_inv)

        if save_as:
            self.save_as = save_as
            self.__save()

        return self.__im

    def point_statistic(self, data):
        """point statistic
        data: point data, [[x,y,number]] or [[x,y]]
        """
        heat_data = np.zeros((self.width, self.height), np.int16)
        for hit in data:
            x, y, n = hit
            heat_data[x, y] += n
        return heat_data

    def __save(self):
        """image save
        """
        save_as = os.path.join(os.getcwd(), self.save_as)
        folder, fn = os.path.split(save_as)
        if not os.path.isdir(folder):
            os.makedirs(folder)

        self.__im.save(save_as)
        self.__im = None

    def get_max_size(self, data):
        """get data max x and max y
        data: point data, [[x,y,number]] or [[x,y]]
        """
        max_w = 0
        max_h = 0

        for hit in data:
            w = hit[0]
            h = hit[1]
            if w > max_w:
                max_w = w
            if h > max_h:
                max_h = h

        return max_w + 1, max_h + 1

    def mk_circle(self, r, w):
        u"""根据半径r以及图片宽度 w ，产生一个圆的list
        @see http://oldj.net/article/bresenham-algorithm/
        r: int, heat radius
        w: int, image width
        """

        # __clist = set()
        __tmp = {}

        def c8(ix, iy, v=1):
            # 8对称性
            ps = (
                (ix, iy),
                (-ix, iy),
                (ix, -iy),
                (-ix, -iy),
                (iy, ix),
                (-iy, ix),
                (iy, -ix),
                (-iy, -ix),
            )
            for x2, y2 in ps:
                p = w * y2 + x2
                __tmp.setdefault(p, (v, x2, y2))

        # 中点圆画法
        x = 0
        y = max(x, r)
        d = 3 - (r << 1)
        while x <= y:
            for _y in range(x, y + 1):
                c8(x, _y, y + 1 - _y)
            if d < 0:
                d += (x << 2) + 6
            else:
                d += ((x - y) << 2) + 10
                y -= 1
            x += 1

        return __tmp.items()

    def mk_colors(self, n=240):
        u"""生成色盘
        @see http://oldj.net/article/heat-map-colors/

        TODO: 根据 http://oldj.net/article/hsl-to-rgb/ 将 HSL 转为 RGBA
        """

        colors = []
        n1 = int(n * 0)
        n2 = n - n1

        for i in range(n1):
            color = "hsl(240, 100%%, %d%%)" % (100 * (n1 - i / 2) / n1)
            colors.append(color)
        for i in range(n2):
            color = "hsl(%.0f, 100%%, 50%%)" % (self.max_color * (1 - float(i) / n2))
            colors.append(color)
        return colors

    def is_num(self, v):
        u"""判断是否为数字，兼容Py2/Py3"""

        if type(v) in (int, float):
            return True

        if ("%d" % v).isdigit():
            # 兼容Py2的long类型
            return True

        return False


def etest():
    u"""测试方法"""

    data = np.random.randint(0, 1000, (100, 2))
    print("painting..")

    hm = HeatMap(data)
    image = hm.heatmap(r=50)
    cv2.imshow("image_heatmap", image)
    cv2.waitKey()
    print("done.")


if __name__ == "__main__":
    etest()
