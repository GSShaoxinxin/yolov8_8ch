# num_to_color.py
import cv2


def Align(image,M):#执行对图像的平移变换
    alidned_image= cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return alidned_image

_color_map = {
    1: "blue",
    2: "green",
    3: "red",
    4: "rededge",
    5: "nir"
}

# 反向映射
_reverse_color_map = {v: k for k, v in _color_map.items()}

def num_to_color(num):
    """
    将数字 1-5 映射为颜色字符串。
    """
    if num not in _color_map:
        raise ValueError("输入必须是1到5之间的整数")
    return _color_map[num]

def color_to_num(color):
    """
    将颜色字符串映射为数字 1-5。
    """
    color = color.lower()
    if color not in _reverse_color_map:
        raise ValueError(f"不支持的颜色名称: {color}")
    return _reverse_color_map[color]

#
# # 示例用法（可删除）
# if __name__ == "__main__":
#     for i in range(1, 6):
#         color = num_to_color(i)
#         print(f"{i} -> {color} -> {color_to_num(color)}")
