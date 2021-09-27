from .meyersmethod import np, la, update_Ainv, complexarray

m = 100
n = 20
A_ = np.random.rand(m * n).reshape(m, n) + 1j * np.random.rand(m * n).reshape(m, n)
A_ = A_.view(complexarray)
Ainv_ = la.pinv(A_).view(complexarray)

# test 1
if 1:
    n_rand = 3
    print(f"validate with {n_rand} random updates")
    for i in range(n_rand):
        c = np.random.rand(m) + 1j * np.random.rand(m)
        c = c.view(complexarray)
        d = np.random.rand(n) + 1j * np.random.rand(n)
        d = d.view(complexarray)

        update_Ainv(A_, Ainv_, c, d, debug=True)

if 1:
    print(f"Test after rank reduction")

    uA, sA, vAh = la.svd(A_, full_matrices=False)

    sAt = sA.copy()
    sAt[-5:] = 0  # remove the last eigenvalues

    At = uA @ np.diag(sAt) @ vAh
    Atinv = la.pinv(At).view(complexarray)

    c = np.random.rand(m) + 1j * np.random.rand(m)
    c = c.view(complexarray)
    d = np.random.rand(n) + 1j * np.random.rand(n)
    d = d.view(complexarray)

    update_Ainv(At, Atinv, c, d, debug=True)

if 1:
    print(f"Test suggested by Cyrille")
    
    m = 4
    n = 4
    
    u1 = np.array([1, 1, 1, 1]).view(complexarray)
    u2 = np.array([1, -1, 1, -1]).view(complexarray)
    A = np.outer(u1, u1.H) + np.outer(u2, u2.H)
    Ainv = la.pinv(A).view(complexarray)
    
    update_Ainv(A, Ainv, np.random.rand(m), np.random.rand(n))
    
    