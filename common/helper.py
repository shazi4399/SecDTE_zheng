import random
from common.constants import *
import hashlib
from common.completeBinaryTree import BTNode
import math

class PRF:
    """A pseudorandom function (PRF).
    A PRF is determined by a secret key and a public maximum.
    """
    def __init__(self, key, bound):
        """Create a PRF determined by the given key and (upper) bound.

        The key is given as a byte string.
        Output values will be in range(bound).
        """
        self.key = key
        self.max = bound
        self.byte_length = ((bound - 1).bit_length() + 7) // 8
        if bound & (bound - 1):  # no power of 2
            self.byte_length += len(self.key)

    def __call__(self, s, n=None):
        """Return a number or length-n list of numbers in range(self.max) for input bytes s."""
        if n == 0:
            return []

        n_ = 1 if n is None else n
        l = self.byte_length
        if not l:
            x = [0] * n_
        else:
            dk = hashlib.pbkdf2_hmac('sha1', self.key, s, 1, n_ * l)
            byteorder = 'little'
            from_bytes = int.from_bytes  # cache
            bound = self.max
            x = [from_bytes(dk[i:i + l], byteorder) % bound for i in range(0, n_ * l, l)]
        return x[0] if n is None else x

def vec_share(A,bound):
    size = len(A)
    r1,r2 = [],[]
    for i in range(size):
        r1.append(random.randint(0, bound))
        v = A[i] - r1[i]
        while v<0:
            v += bound
        r2.append(v)
    return [r1, r2]

def vec2mat_share(A,bound):
    # 检查输入是否为列表
    if not isinstance(A, list):
        raise ValueError("输入必须是一个列表")
        # 创建n*1的矩阵
    matrix = [[element] for element in A]
    len_size = len(matrix)
    width_size = len(matrix[0])
    m1 = [[] for i in range(len_size)]
    m2 = [[] for i in range(len_size)]
    for i in range(len_size):
        for j in range(width_size):
            m1[i].append(random.randint(0,bound))
            v = matrix[i][j] - m1[i][j]
            while v < 0:
                v += bound
            m2[i].append(v)
    return [m1,m2]
def mat_share(M,bound):
    len_size = len(M)
    width_size = len(M[0])
    m1 = [[] for i in range(len_size)]
    m2 = [[] for i in range(len_size)]
    for i in range(len_size):
        for j in range(width_size):
            m1[i].append(random.randint(0,bound-1))
            v = M[i][j] - m1[i][j]
            while v < 0:
                v += bound
            m2[i].append(v)
    return [m1,m2]

def generate_random_matrix(rows, cols, bound):
    matrix = []
    for _ in range(rows):
        row = [random.randint(0, bound-1) for _ in range(cols)]  # 生成每一行的随机数
        matrix.append(row)
    return matrix

def vec_sub(A,B):
    size = len(A)
    C=[]
    for i in range(size):
        v = A[i] - B[i]
        if v<0:
            v += VEC_VAL_MAX_BOUND
        C.append(v)
    return C

def vec_add(A, B):
    size = len(A)
    C = []
    for i in range(size):
        v = (A[i] + B[i]) % VEC_VAL_MAX_BOUND
        C.append(v)
    return C

def vec_add_withBound(A, B, bound):
    size = len(A)
    C = []
    for i in range(size):
        v = (A[i] + B[i]) % bound
        C.append(v)
    return C

def vec_sub_withBound(A, B, bound):
    size = len(A)
    C = []
    for i in range(size):
        v = A[i] - B[i]
        if v<0:
            v+=bound
        C.append(v)
    return C

def int_2of2_share(a,bound):
    # 设置随机种子为固定值，比如123
    random.seed(123)
    r1 = random.randint(0, bound-1)
    r2 = a - r1
    while r2 < 0:
        r2 += bound
    return [r1, r2]

def int_share(a,bound):
    r1 = random.randint(0, bound-1)
    r2 = a - r1
    while r2 < 0:
        r2 += bound
    return [r1, r2]

def circ_shift(A, r):
    circLen = len(A)
    B = A.copy()
    for i in range(circLen):
        B[(i+r)%circLen] = A[i]
    return B

def vec_inverse(A):
    size = len(A)
    B = []
    for i in range(size):
        B.append(VEC_VAL_MAX_BOUND - A[i])
    return B

def vec_array_sub(A,B):
    size = len(A)
    C=[]
    for i in range(size):
        C.append(vec_sub(vec_inverse(A[i]), B[i]))
    return C

def permuat_vector(A,r):
    circLen = len(A)
    B = A.copy()
    for i in range(circLen):
        B[(i+r)%circLen] = A[i]
    return B

def invers_permuat_vector(A,r):
    circLen = len(A)
    B = A.copy()
    for i in range(circLen):
        B[(i-r)%circLen] = A[i]
    return B


def replicated_local_add(A, B):
    return [vec_add(A[0], B[0]),  vec_add(A[1], B[1])]

def replicated_local_sub(A, B):
    return [vec_sub(A[0], B[0]),  vec_sub(A[1], B[1])]

def ass_local_mul(A, B):
    return
def toReplicatedConverse(v1,v2,bound):
    size = len(v1)
    delta1 = []
    delta2 = []
    for i in range(size):
        delta1.append(random.randint(0, bound))
        delta2.append(random.randint(0, bound))
    r1 = vec_add_withBound(delta2, delta1, bound)
    r2 = vec_sub_withBound(v1, delta1, bound)
    r3 = vec_sub_withBound(v2, delta2, bound)
    return [[r1, r2], [r2, r3], [r3, r1]]

def recons_2of3Shares(A,bound):
    r1 = A[0][0]
    r2 = A[0][1]
    r3 = A[1][1]
    out = (r1+r2+r3) % bound
    return out

def inRing(val,bound):
    ret = None
    if val<0:
        ret = val + bound
    else:
        ret = val % bound
    return ret

def ASS_local_add(A,B,bound):
    return (A+B)%bound

def ASS_local_mul(A, scaler, bound):
    return (A*scaler)%bound

def RSS_local_add(A,B,bound):
    return [(A[0]+ B[0])%bound,  (A[1]+ B[1])%bound]

def RSS_local_mul(A,scaler,bound):
    return [(A[0]*scaler)%bound,  (A[1]*scaler)%bound]

def constructCompleteTreeFromList(b_pi):
    allNodes = []
    #Firstly, create these nodes
    for v in b_pi:
        allNodes.append(BTNode(None, v))

    #Secondly,construct relation between nodes
    nonLeafMax = math.floor(len(b_pi)/2)
    for i, v in enumerate(b_pi):
        if i < nonLeafMax:
            curNode = allNodes[i]
            leftNode = allNodes[(i+1)*2-1]
            rightNode = allNodes[(i+1)*2]
            curNode.setLeftAs(leftNode)
            curNode.setRightAs(rightNode)
    return allNodes

def shuffleNonLeaf(nodes, b_pi):
    # From wideth priority list to a tree
    treeNodes = constructCompleteTreeFromList(nodes)
    piNodes = constructCompleteTreeFromList(b_pi)

    ret=[]
    piQueue = [piNodes[0]]
    dataQueue = [treeNodes[0]]
    while len(piQueue) > 0:
        topPi = piQueue.pop(0)
        topVal = dataQueue.pop(0)
        flipVal = topPi.getVal()
        leftChild = topVal.getLeft()
        if leftChild is not None:
            if flipVal:
                topVal.flipLeftRight()
            dataQueue.append(topVal.getLeft())
            dataQueue.append(topVal.getRight())
            piQueue.append(topPi.getLeft())
            piQueue.append(topPi.getRight())
        ret.append(topVal.getVal())
    return ret

def inversePermutation(b_pi):
    allNodes = constructCompleteTreeFromList(b_pi)

    #Thirdly, execute a wideth priority iteration with flipping values
    inverse=[]
    queue = [allNodes[0]]
    while len(queue) > 0:
        top = queue.pop(0)
        flipVal = top.getVal()
        leftChild = top.getLeft()
        if leftChild is not None:
            if flipVal:
                top.flipLeftRight()
            queue.append(top.getLeft())
            queue.append(top.getRight())
        inverse.append(flipVal)
    return inverse

def permute(b_pi,vec):
    vec_len = len(vec)
    # assert len(b_pi)+1 == vec_len
    height=1
    for i,v in enumerate(b_pi):
        interval = int(vec_len / 2**height)
        if v:
            ll = (i+1 - 2**(height-1))*(interval*2)
            lr=ll+interval
            rl=lr
            rr = rl+interval
            # print("bounds are: ",ll,lr,rl,rr)
            temp = vec[ll:lr].copy()
            vec[ll:lr] = vec[rl:rr]
            vec[rl:rr] = temp
        if i == 2**height -2:
            height+=1
    return vec

def getEvalIndex(nonLeafNodes):
    interval=len(nonLeafNodes)+1
    curHeight,index=0,1
    maxHeight = int(math.log(interval,2))
    offset=0
    while curHeight<maxHeight:
        interval = int(interval/2)
        if nonLeafNodes[index-1]==1:
            index=2*index+1
            offset += interval
        else:
            index = 2*index
        curHeight+=1
    # print(curHeight)
    # print(maxHeight)
    return offset

def one_hot_encode(i,nDim):
    print("i:",i)
    one_hot = [0]*nDim
    one_hot[i] = 1
    return one_hot


def matrix_multiply(A, B):
    # 获取矩阵A和B的行数和列数
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    # 检查矩阵A的列数是否等于矩阵B的行数
    if cols_A != rows_B:
        raise ValueError("矩阵A的列数必须等于矩阵B的行数")

        # 创建一个结果矩阵，其行数为矩阵A的行数，列数为矩阵B的列数
    C = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    # 遍历矩阵A和B的元素，进行乘法运算和累加
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += (A[i][k] * B[k][j])%pow(2,BITS_REPRESENTED)
    return C


def matrix_addition(A, B):
    # 检查两个矩阵是否具有相同的形状
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("两个矩阵的形状必须相同")

        # 创建一个新的矩阵来存储结果
    C = [[0 for _ in range(len(A[0]))] for _ in range(len(A))]
    # 遍历矩阵A和B的元素，并将它们相加
    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i][j] = (A[i][j] + B[i][j])%pow(2,BITS_REPRESENTED)
    return C

def matrix_subtraction(A, B):
    # 检查两个矩阵是否具有相同的形状
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("两个矩阵的形状必须相同")

        # 创建一个新的矩阵来存储结果
    C = [[0 for _ in range(len(A[0]))] for _ in range(len(A))]
    # 遍历矩阵A和B的元素，并将它们相加
    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i][j] = (A[i][j] - B[i][j])%pow(2,BITS_REPRESENTED)
    return C

def decimal_to_binary(decimal_val):
    bits = []
    while decimal_val > 0:
        bits.append(decimal_val % 2)
        decimal_val //= 2
    return bits


def decimal_to_binary_32(decimal_val):
    bits = []
    # 添加符号位
    decimal_val2 = decimal_val
    if decimal_val < 0:
        decimal_val = abs(decimal_val)
    # 转换为二进制表示
    while decimal_val > 0:
        bits.append(decimal_val % 2)
        decimal_val //= 2

    # 补充零使二进制表示长度达到31位（因为已经有了符号位）
    while len(bits) < BINARY_REPRESENTED-1:
        bits.append(0)
    print(bits)
    # 如果是负数，取补码形式
    if decimal_val2 < 0:
        # 取反
        bits = [1 if bit == 0 else 0 for bit in bits]
        # 加1
        carry = 1
        for i in range(BINARY_REPRESENTED-1):
            bits[i] += carry
            if bits[i] > 1:
                bits[i] = 0
            else:
                carry = 0
        print(bits)
    # 添加符号位
    if decimal_val2 < 0:
        bits.append(1)
    else:
        bits.append(0)
    return bits
    # 返回32位二进制列表
    # 返回32位二进制列表（注意要反转列表，使得低位在前，高位在后）
    #return bits[::-1]

#
# # 示例用法
# binary_32 = decimal_to_binary_32(-7)
# print(binary_32)
