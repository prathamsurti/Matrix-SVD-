import numpy as np


def create_matrix(n) : 
    l=[]
    for i in range(n):
        r=[]
        for j in range(n):
            value=float(input(f"Enter A{i}{j} =  "))
            r.append(value)
        l.append(r)



    return np.array(l,dtype=float)


def SVD_Decomposition(l):
    U,S,VT = np.linalg.svd(l,full_matrices=False)
    return U,S,VT 

def Reconstruction(U,S,VT): 
    Sigma = np.diag(S)
    A_reconstructed = np.dot(U, np.dot(Sigma, VT))
    return A_reconstructed

def Low_rankApproximation(U,S,VT,k): 

    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]
    A_low_rank = np.dot(U_k, np.dot(S_k, VT_k))
    return A_low_rank 



n= int(input("Enter size of the matrix : "))

A = create_matrix(n)


print(f"Original Matrix \n {A}")

U,S,VT = SVD_Decomposition(A)

print(f'U(left singular Vectors):\n{U} \n S(singular Values):\n {S} \n VT(Right Singular vectors): \n {VT}')

A_reconstructed = Reconstruction(U,S,VT)

print(f'Reconstructed Matrix: \n {A_reconstructed}')


k = int(input("\n Enter the rank for the low-rank approximation : "))

A_low_rank= Low_rankApproximation(U,S,VT,k)


print(A_low_rank)