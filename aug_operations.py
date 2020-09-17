import numpy as np
import cv2


def convert_coords(pts, coef, intercept):
    """
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    coef = M[:, :2].T.astype('float32')
    intercept = M[:, 2:3].T.astype('float32')

    :param pts:         np.ndarray (None, 2)
    :param coef:        np.ndarray (2, 3)
    :param intercept:   np.ndarray (1, 3)
    :return:            np.ndarray (None, 2)
    """
    new_pts = pts @ coef + intercept
    new_pts = new_pts[:, :2] / new_pts[:, 2:3]
    return new_pts


def resize_and_augment(image, json_label=None, new_h=None, new_w=None, flip=True, flip_prob=0.5, gray=False,
                       gray_prob=0.5, min_mini_scale=0.7, max_mini_scale=1.3, side_shift=0.05, color_temp_shift=10,
                       bright_shift=20, gamma=1.2, angle_max=7, noise_max=7):
    """
    :param image:              3d channel-last bgr numpy.ndarray (h,w,ch) in uint8
    :param json_label:         bool
    :param new_h:              int, new image high
    :param new_h:              int, new image width
    :param flip:               bool
    :param flip_prob:          float in [0, 1]
    :param gray:               bool
    :param gray_prob:          float in [0, 1]
    :param min_mini_scale:     float in (0, max_scale)
    :param max_mini_scale:     float in (min_scale, inf)
    :param side_shift:         float in [0, 1]
    :param color_temp_shift:   float/integer in [0, 255]
    :param bright_shift:       float/integer in [0, 255]
    :param gamma:              float in (0, inf)
    :param angle_min           int in (-45, angle_max)
    :param angle_max           int in (angle_min, 45)
    :param noise_max           int in (0,255)
    :return:
    """

    # white noise
    h, w = image.shape[:2]
    noise_max_random = np.random.uniform(0, noise_max)
    noise = np.random.laplace(0., noise_max_random, size=(h, w, 3)).astype('float32')
    image = image.astype('float32') + noise
    image = np.clip(image, 0, 255)
    image = image.astype("uint8")

    # resize
    new_h = h if new_h is None else new_h
    new_w = w if new_w is None else new_w
    M = np.identity(3, 'float64')
    M[0, 0] = new_h / h
    M[1, 1] = new_w / w

    # 旋转
    h, w = new_h, new_w
    angle = np.random.uniform(low=-angle_max, high=angle_max)
    _M = np.zeros((3, 3), 'float64')
    _M[:2, :] = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
    _M[2, 2] = 1
    # M = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
    # image = cv2.warpAffine(image, M, (w, h))

    # 对图像做适当的缩小或放大
    # 对图像做适当的缩小或放大
    scale = 1.
    scale_adj = np.random.uniform(min_mini_scale, max_mini_scale)
    new_h, new_w = h * scale * scale_adj, w * scale * scale_adj

    # 映射图上做细微挪动的拉伸变换
    this_side_shift = min(abs(1. / scale_adj - 1.), side_shift)

    dx_up = np.random.uniform(0, this_side_shift) * new_w
    dx_down = np.random.uniform(0, this_side_shift) * new_w
    dy_left = np.random.uniform(0, this_side_shift) * new_h
    dy_right = np.random.uniform(0, this_side_shift) * new_h

    src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst_pts = np.array(
        [
            [0 + dx_up, 0 + dy_left],
            [new_w + dx_up, 0 + dy_right],
            [new_w + dx_down, new_h + dy_right],
            [0 + dx_down, new_h + dy_left]
        ],
        dtype=np.float32)

    # 原图上做中心移动
    cx = np.random.uniform(low=0, high=max(0, np.max(dst_pts[:, 0]) - new_w) / (scale * scale_adj))
    cy = np.random.uniform(low=0, high=max(0, np.max(dst_pts[:, 1]) - new_h) / (scale * scale_adj))
    src_pts[:, 0] += cx
    src_pts[:, 1] += cy

    _M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M = _M @ M
    dst_img = cv2.warpPerspective(image, M, (int(new_w), int(new_h)))

    # 水平翻转
    flip = np.random.choice([True, False], 1, p=[flip_prob, 1. - flip_prob])[0] if flip else flip
    if flip:
        dst_img = dst_img[:, ::-1, :]

    if json_label is not None:
        coef = M[:, :2].T.astype('float32')
        intercept = M[:, 2:3].T.astype('float32')
        for label_i in range(len(json_label['shapes'])):
            pts = np.array(json_label['shapes'][label_i]['points'], 'float32')
            pts_new = convert_coords(pts, coef, intercept)
            if flip:
                pts_new[:, 0] = dst_img.shape[1] - pts_new[:, 0]
            json_label['shapes'][label_i]['points'] = pts_new.tolist()

    # 色温变换
    temp_shift = np.random.uniform(-color_temp_shift, color_temp_shift, (1, 1, 3)).astype('float32')
    # 亮度变换
    b_shift = np.random.uniform(-bright_shift, bright_shift)
    dst_img = (dst_img.astype('float32') + (temp_shift + b_shift)).clip(0, 255)
    # gamma对比度变换
    _gamma = abs(1. - gamma)
    g_shift = np.random.uniform(-_gamma, _gamma) + 1.
    dst_img = (np.power(dst_img / 255., g_shift) * 255.).astype('uint8')
    # 转灰度
    gray = np.random.choice([True, False], 1, gray_prob)[0] if gray else gray
    if gray:
        dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)
        dst_img = cv2.cvtColor(dst_img, cv2.COLOR_GRAY2BGR)

    if json_label is None:
        return dst_img
    else:
        return dst_img, json_label


def mapping_image_size(image, mapping_size_min=32, mapping_size_max=70, target_size_min=32, target_size_max=64):
    """
    mapping image to (target_size_min,target_size_max)
    :param image:             3d channel-last bgr numpy.ndarray (h,w,ch) in uint8
    :param mapping_size_min:  int
    :param mapping_size_max:  int, max than mapping_size_min
    :param target_size_min:   int
    :param target_size_max:   int
    :return:
    """
    img = image.copy()
    H, W = img.shape[:2]

    if H > W:
        resize_h = max(mapping_size_min, H)
        resize_h = min(mapping_size_max, resize_h)
        resize_h = int(
            (resize_h - mapping_size_min) / (mapping_size_max - mapping_size_min) * (
                    target_size_max - target_size_min) + target_size_min)
        img = cv2.resize(img, (int(W * resize_h / H), resize_h))
    else:
        resize_w = max(mapping_size_min, W)
        resize_w = min(mapping_size_max, resize_w)
        resize_w = int(
            (resize_w - mapping_size_min) / (mapping_size_max - mapping_size_min) * (
                    target_size_max - target_size_min) + target_size_min)
        img = cv2.resize(img, (resize_w, int(H * resize_w / W)))
    return img


def resize_and_padding_black(image, new_h, new_w):
    """
    image padding with black to specified size.
    :param image:         3d channel-last bgr numpy.ndarray (h,w,ch) in uint8
    :param new_h:         int
    :param new_w:         int
    :return:
    """
    img = image.copy()
    H, W = img.shape[:2]
    scale = max(H / new_h, W / new_w)
    resize_W = int(W / scale)
    resize_H = int(H / scale)
    img = cv2.resize(img, (resize_W, resize_H))
    pad_w = max(new_w - resize_W, 0)
    pad_h = max(new_h - resize_H, 0)
    img = cv2.copyMakeBorder(img, int(pad_h / 2), pad_h - int(pad_h / 2), int(pad_w / 2),
                             pad_w - int(pad_w / 2), cv2.BORDER_CONSTANT)
    return img


def crop_and_pad(img, new_h, new_w):
    h, w = img.shape[:2]
    img = cv2.copyMakeBorder(
        img,
        top=0, bottom=max(0, new_h - h), left=0, right=max(0, new_w - w),
        borderType=cv2.BORDER_CONSTANT
    )[:new_h, :new_w]
    return img


if __name__ == "__main__":
    img = cv2.imread("/home/chenli/formal_work/claim/data/test_image/票据、清单及出院小结/省肿瘤医院/出院小结.jpg")
    cv2.imshow("origin", img)
    img = resize_and_padding_black(img, 960, 960)
    cv2.imshow("img", img)
    cv2.waitKey()
