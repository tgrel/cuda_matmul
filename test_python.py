import numpy as np

import cuda_matmul

def matmul(A, B):

    A = A.astype(np.float32)
    B = B.astype(np.float32)
    
    R = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
    lhs = A.__array_interface__['data'][0]
    rhs = B.__array_interface__['data'][0]
    result = R.__array_interface__['data'][0]

    print lhs, rhs, result

    cuda_matmul.matmul(lhs, A.shape[0], A.shape[1],
                       rhs, B.shape[0], B.shape[1],
                       result)

    return R



test_data = [
    (np.random.uniform(size=(3,5)), np.random.uniform(size=(5,2))),
    (np.random.uniform(size=(3,5)), np.random.uniform(size=(5,2))),
    (np.random.uniform(size=(20,30)), np.random.uniform(size=(30,2))),
    (np.random.uniform(size=(1,10000)), np.random.uniform(size=(10000,1))),
    (np.random.uniform(size=(1000,1000)), np.random.uniform(size=(1000,1000)))
    ]

for lhs, rhs in test_data:
    lhs = lhs.astype(np.float32)
    rhs = rhs.astype(np.float32)
    
    cuda_result = matmul(lhs, rhs)
    reference_result  = lhs.dot(rhs)
    print cuda_result - reference_result




