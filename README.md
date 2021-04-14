# xyz2expmap
human3.6m格式下將xyz坐标转化为李代数（可能会有bug？）

Transforming XYZ coordinates into Lie Algebras(expmap) in human3.6m format(There may be some bugs in the code)

使用方法：
from xyz2expmap import xyz2expmap
'''
输入是32×3的矩阵，输出33×3的矩阵
'''
expdata = xyz2expmap(xyzdata)
