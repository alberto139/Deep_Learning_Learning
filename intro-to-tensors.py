import torch


# Create a couple of tensors
A = [[1, 2, 1],[0, 1, 0],[2, 3, 4]]
tensor_A = torch.tensor(A)

A = [[2, 5, 1],[6, 7, 1],[1, 8, 1]]
tensor_B = torch.tensor(A)

# Element Wise Multiplication
print(tensor_A * tensor_B)
"""
tensor([[ 2, 10,  1],
        [ 0,  7,  0],
        [ 2, 24,  4]])
"""

# Matrix Multiplication (Dot product)
print(tensor_A @ tensor_B)
"""
tensor([[15, 27,  4],
        [ 6,  7,  1],
        [26, 63,  9]])
"""

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")