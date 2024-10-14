import math

def parse_equation(equation):
    equation = equation.replace(" ", "")
    left_side, right_side = equation.split('=')
    
    x_coeff = y_coeff = z_coeff = 0
    
    terms = left_side.replace('-', '+-').split('+')
    
    for term in terms:
        if 'x' in term:
            coef = term.replace('x', '')
            if coef == "-":
                coef = -1
            x_coeff = int(coef or '1')
        elif 'y' in term:
            coef = term.replace('y', '')
            if coef == "-":
                coef = -1
            y_coeff = int(coef or '1')
        elif 'z' in term:
            coef = term.replace('z', '')
            if coef == "-":
                coef = -1
            z_coeff = int(coef or '1')
    
    constant = int(right_side)
    
    return [x_coeff, y_coeff, z_coeff], constant

def read_system_of_equations(file_path):
    A = []
    B = []
    
    with open(file_path, 'r') as file:
        for line in file:
            coefficients, constant = parse_equation(line.strip())
            A.append(coefficients)
            B.append(constant)
    
    return A, B

def determinant_3x3(A):
    if len(A) != 3 or any(len(row) != 3 for row in A):
        raise ValueError("The matrix must be 3x3")
    
    a11, a12, a13 = A[0]
    a21, a22, a23 = A[1]
    a31, a32, a33 = A[2]
    
    det = (a11 * (a22 * a33 - a23 * a32) 
           - a12 * (a21 * a33 - a23 * a31) 
           + a13 * (a21 * a32 - a22 * a31))
    
    return det

def trace_3x3(A):
    if len(A) != 3 or any(len(row) != 3 for row in A):
        raise ValueError("The matrix must be 3x3")
    
    trace = A[0][0] + A[1][1] + A[2][2]
    
    return trace

def euclidean_norm(B):
    if len(B) != 3:
        raise ValueError("The vector must have 3 elements")
    
    norm = math.sqrt(B[0]**2 + B[1]**2 + B[2]**2)
    
    return norm

def transpose_3x3(A):
    if len(A) != 3 or any(len(row) != 3 for row in A):
        raise ValueError("The matrix must be 3x3")
    
    A_T = [[A[j][i] for j in range(3)] for i in range(3)]
    
    return A_T

def matrix_vector_multiply(A, B):
    if len(A) != 3 or any(len(row) != 3 for row in A):
        raise ValueError("The matrix must be 3x3")
    if len(B) != 3:
        raise ValueError("The vector must have 3 elements")
    
    C = [0, 0, 0]
    
    for i in range(3):
        C[i] = A[i][0] * B[0] + A[i][1] * B[1] + A[i][2] * B[2]
    
    return C

def cramer_solve(A, B):
    det_A = determinant_3x3(A)
    
    if det_A == 0:
        raise ValueError("The determinant of matrix A is 0. The system has no unique solution.")
    
    A1 = [[B[0], A[0][1], A[0][2]], [B[1], A[1][1], A[1][2]], [B[2], A[2][1], A[2][2]]]  # Replace first column of A with B
    A2 = [[A[0][0], B[0], A[0][2]], [A[1][0], B[1], A[1][2]], [A[2][0], B[2], A[2][2]]]  # Replace second column of A with B
    A3 = [[A[0][0], A[0][1], B[0]], [A[1][0], A[1][1], B[1]], [A[2][0], A[2][1], B[2]]]  # Replace third column of A with B
    
    det_A1 = determinant_3x3(A1)
    det_A2 = determinant_3x3(A2)
    det_A3 = determinant_3x3(A3)
    
    x1 = det_A1 / det_A
    x2 = det_A2 / det_A
    x3 = det_A3 / det_A
    
    return [x1, x2, x3]

def determinant_2x2(M):
    return M[0][0] * M[1][1] - M[0][1] * M[1][0]

def cofactor_matrix(A):
    cofactor = []
    for i in range(3):
        cofactor_row = []
        for j in range(3):
            minor = [
                [A[m][n] for n in range(3) if n != j]
                for m in range(3) if m != i
            ]
            cofactor_ij = (-1) ** (i + j) * determinant_2x2(minor)
            cofactor_row.append(cofactor_ij)
        cofactor.append(cofactor_row)
    return cofactor

def inverse_matrix(A):
    det_A = determinant_3x3(A)
    if det_A == 0:
        raise ValueError("The determinant of matrix A is 0. The matrix is not invertible.")
    
    cofactor = cofactor_matrix(A)
    
    adjugate = transpose_3x3(cofactor)
    
    inverse = [[adjugate[i][j] / det_A for j in range(3)] for i in range(3)]
    
    return inverse

file_path = 'input.txt' 
A, B = read_system_of_equations(file_path)
print("Matrix A (coefficients):", A)
print("Vector B (constants):", B)

det_A = determinant_3x3(A)
print("Determinant of matrix A:", det_A)

trace_A = trace_3x3(A)
print("Trace of matrix A:", trace_A)

norm_B = euclidean_norm(B)
print("Euclidean norm of vector B:", norm_B)

transpose_A = transpose_3x3(A)
print("Transpose of matrix A:", transpose_A)

result = matrix_vector_multiply(A, B)
print("Matrix-vector multiplication result:", result)

solution = cramer_solve(A, B)
print("Solution to the system (x, y, z): [{:.2f}, {:.2f}, {:.2f}]".format(solution[0], solution[1], solution[2]))


A_inv = inverse_matrix(A)

X = matrix_vector_multiply(A_inv, B)

print("Solution to the system (x, y, z): [{:.2f}, {:.2f}, {:.2f}]".format(X[0], X[1], X[2]))


