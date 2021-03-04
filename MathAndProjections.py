import numpy as np

def CKM_to_CKH(vertex, origin):
    rows, cols = vertex.shape

    #adds col of ones to right side of matrix
    vertex_ex = np.c_[vertex, np.ones(rows)]

    T = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [-origin[0], -origin[1], -origin[2], 1]])

    S = np.array([[-1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    R_90x = np.array([[1, 0, 0, 0],
                      [0, 0, -1, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]])

    d = np.sqrt(origin[0] ** 2 + origin[1] ** 2)

    if d != 0:
        R_uy = np.array([[origin[1] / d, 0, origin[0] / d, 0],
                         [0, 1, 0, 0],
                         [-origin[0] / d, 0, origin[1] / d, 0],
                         [0, 0, 0, 1]])
    else:
        R_uy = np.eye(4)

    s = np.sqrt(d**2 + origin[2] ** 2)

    if s != 0:
        R_wx = np.array([[1, 0, 0, 0],
                         [0, d / s, -origin[2] / s, 0],
                         [0, origin[2] / s, d / s, 0],
                         [0, 0, 0, 1]])
    else:
        R_wx = np.eye(4)

    V = T @ S @ R_90x @ R_uy @ R_wx

    return (vertex_ex @ V)[:, :cols], s


def perspective_proj(vertex, s):
    vertexCKK = vertex

    for point in vertexCKK:
        if point[2] != 0:
            point[0] *= s / point[2]
            point[1] *= s / point[2]

    return vertexCKK[:, 0:2]

def parallel_proj(vertex, s):
    return vertex[:, 0:2]


def CKK_to_CKEi(vertex, pk, xc, yc, xe, ye):
    for point in vertex:
        point[0] *= xe / pk
        point[1] *= -ye / pk
        point[0] += xc
        point[1] += yc

    return vertex

def plane_coef(face, all_edges):

    e = np.array([all_edges[face[i]] for i in range(3)])

    A = (e[2][1] - e[0][1]) * (e[1][2] - e[0][2]) - (e[1][1] - e[0][1]) * (e[2][2] - e[0][2])
    B = (e[1][0] - e[0][0]) * (e[2][2] - e[0][2]) - (e[2][0] - e[0][0]) * (e[1][2] - e[0][2])
    C = (e[2][0] - e[0][0]) * (e[1][1] - e[0][1]) - (e[1][0] - e[0][0]) * (e[2][1] - e[0][1])
    D = -(A * e[0][0] + B * e[0][1] + C * e[0][2])

    return np.array([A, B, C, D])

def plane_w_center(all_edges):
    edges = np.array(all_edges)
    return np.apply_along_axis(sum, 0, edges) / edges.shape[0]

def matrix_to_w_center(p_coef, w_center):
    return p_coef if p_coef[0] * w_center[0] + p_coef[1] * w_center[1] + p_coef[2] * w_center[2] + p_coef[3] < 0 else -p_coef
