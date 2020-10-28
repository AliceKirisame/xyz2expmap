import math
import numpy
np = numpy
#from data_utils import rotmat2expmap, expmap2rotmat

def rotmat2expmap(R):
    theta = np.arccos((np.trace(R) - 1) / 2.0)
    if theta < 1e-6:
        A = np.zeros((3, 1))
    else:
        A = theta / (2 * np.sin(theta)) * np.array([[R[2, 1] - R[1, 2]], [R[0, 2] - R[2, 0]], [R[1, 0] - R[0, 1]]], dtype=object)

    return A.squeeze()


def expmap2rotmat(A):
    theta = np.linalg.norm(A)
    if theta == 0:
        R = np.identity(3)
    else:
        A = A / theta
        cross_matrix = np.array([[0, -A[2], A[1]], [A[2], 0, -A[0]], [-A[1], A[0], 0]], dtype=object)
        R = np.identity(3) + np.sin(theta) * cross_matrix + (1 - np.cos(theta)) * np.matmul(cross_matrix, cross_matrix)

    return R


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def normalize(a):
    a = numpy.array(a)
    return numpy.sqrt(numpy.sum(numpy.power(a, 2)))

def cal_rotate_matrix(a, b):
    rot_axis = cross(b, a)
    rot_angle = math.acos(dot(a, b) / normalize(a) / normalize(b))

    norm = normalize(rot_axis)
    rot_mat = numpy.zeros((3, 3), dtype = "float32")

    rot_axis = (rot_axis[0] / norm, rot_axis[1] / norm, rot_axis[2] / norm)

    rot_mat[0, 0] = math.cos(rot_angle) + rot_axis[0] * rot_axis[0] * (1 - math.cos(rot_angle))
    rot_mat[0, 1] = rot_axis[0] * rot_axis[1] * (1 - math.cos(rot_angle)) - rot_axis[2] * math.sin(rot_angle)
    rot_mat[0, 2] = rot_axis[1] * math.sin(rot_angle) + rot_axis[0] * rot_axis[2] * (1 - math.cos(rot_angle))

    rot_mat[1, 0] = rot_axis[2] * math.sin(rot_angle) + rot_axis[0] * rot_axis[1] * (1 - math.cos(rot_angle))
    rot_mat[1, 1] = math.cos(rot_angle) + rot_axis[1] * rot_axis[1] * (1 - math.cos(rot_angle))
    rot_mat[1, 2] = -rot_axis[0] * math.sin(rot_angle) + rot_axis[1] * rot_axis[2] * (1 - math.cos(rot_angle))

    rot_mat[2, 0] = -rot_axis[1] * math.sin(rot_angle) + rot_axis[0] * rot_axis[2] * (1 - math.cos(rot_angle))
    rot_mat[2, 1] = rot_axis[0] * math.sin(rot_angle) + rot_axis[1] * rot_axis[2] * (1 - math.cos(rot_angle))
    rot_mat[2, 2] = math.cos(rot_angle) + rot_axis[2] * rot_axis[2] * (1 - math.cos(rot_angle))

    return numpy.array(rot_mat)

if __name__ == '__main__':
    a = (-0.006576016845720566, 0.20515224329972243, 0.011860567926381188)
    b = (0, 0.2056, 0) 
    rot_mat = cal_rotate_matrix(a, b)
    print(rot_mat)
    expmap = rotmat2expmap(rot_mat)
    rot_mat = expmap2rotmat(expmap)
    print(rot_mat)
    print(b)
    print(numpy.array(a) * numpy.matrix(rot_mat))
