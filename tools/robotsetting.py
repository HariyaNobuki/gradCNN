import numpy as np
import math

"""Link"""
class Link:
    # コンストラクタ初期化
    def __init__(self,angle,length):
        self.angle = math.radians(angle)          # 逆運動学によるとここさえわかれば駆動することになる
        self.length = length
        # 初期更新
        self.x = self.length * np.sin(self.angle)
        self.y = self.length * np.cos(self.angle)


"""Circle"""
class Circle:
    def __init__(self):
        self.R = 8
        self.center_x = 3 # 中心のつもりだけどどうだろうか
        self.center_y = 3