from common.helper import permuat_vector
from common.helper import invers_permuat_vector


def left_shift_vector(A, r):
    circLen = len(A)
    B = A.copy()
    # 计算等效的右移距离
    shift_right = (circLen - r) % circLen
    for i in range(circLen):
        B[(i + shift_right) % circLen] = A[i]
    return B

A = [1,2,3]
B = permuat_vector(A,1)
print(B)
C = invers_permuat_vector(B,1)
print(C)
# 示例使用
