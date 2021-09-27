# -*- coding: utf-8 -*-
"""
Implementation of Meyer's method for rank-one update of the pseudo-inverse
"""
from mathutils import np, la, complexarray, MathError


def update_Ainv(A, Ainv, c, d, debug=False):
    """
    Compute the pseudo-inverse of the matrix A + outer(c, d.H)
    Given Ainv the inverse of A
    This is generally faster than direct computation via numpy.linalg.pinv
    
    If debug is true: the pseudo inverse of the updated A is also directly computed
    and we assert that Meyer's method result is close enough
    """
    assert type(A) is complexarray

    m, n = A.shape
    # test for the six conditions
    c_is_spanned = is_span(c, A)
    d_is_spanned = is_span(d, A.H)
    beta = 1 + d.H @ Ainv @ c
    k = Ainv @ c
    h = d.H @ Ainv
    u = (np.eye(m) - A @ Ainv) @ c
    v = d.H @ (np.eye(n) - Ainv @ A)
    #print(beta)

    if d_is_spanned and beta != 0:
        print("CASE 5")
        uu = u.H.dot(u)
        hh = h.H.dot(h)
        p2 = - uu * Ainv @ h.H / beta.conj() - k
        q2star = - hh * u.H / beta.conj() - h
        sigma2 = hh * uu + beta.conj() * beta
        Aup = (Ainv 
               + np.outer(Ainv @ h.H, u.H) / beta.conj() 
               - beta.conj() * np.outer(p2, q2star) / sigma2)

    elif not c_is_spanned:
        if not d_is_spanned:
            print("CASE 1")
            Aup = (Ainv 
                   - np.outer(k, vec_inv(u)) 
                   - np.outer(vec_inv(v), h) 
                   + beta * np.outer(vec_inv(v), vec_inv(u)))
        elif d_is_spanned and beta == 0:
            print("CASE 4")
            Aup = Ainv - np.outer(Ainv @ vec_inv(h), h) - np.outer(k, vec_inv(u))

    elif c_is_spanned:
        if beta == 0:
            if not d_is_spanned:
                print("CASE 2")
                Aup = Ainv - np.outer(k, vec_inv(k) @ Ainv) - np.outer(vec_inv(v), h)
            else:
                print("CASE 6")
                coef = vec_inv(k) @ Ainv @ vec_inv(h)
                Aup = (Ainv
                       - np.outer(k, vec_inv(k) @ Ainv)
                       - np.outer(Ainv @ vec_inv(h), h)
                       + coef * np.outer(k, h))
        elif beta != 0:
            print("CASE 3")
            kk = k.H.dot(k)
            vv = v.H.dot(v)
            p1 = - kk * v.H / beta.conj() - k
            q1star = - vv * k.H @ Ainv / beta.conj() - h
            sigma1 = kk * vv + beta.conj() * beta
            Aup = (Ainv
                   + np.outer(v.H, k.H @ Ainv) / beta.conj() 
                   - beta.conj() * np.outer(p1, q1star) / sigma1)

    if debug:
        # Warning: this is a heavy computation
        Anew = A + np.outer(c, d.H)
        Anewinv = la.pinv(Anew)
        if not np.allclose(Anewinv, Aup):
            raise MathError(
                f"Meyer's method and direct computation give results differing by norm of {la.norm(Anewinv - Aup):.2e}"
            )
        else:
            print("Meyer's method successful")

    return Aup


def is_span(b, A):
    """
    Test if vector b is in span(A)
    Find if there exists x such that A @ x = b
    """
    x, residuals, rank, s = la.lstsq(A, b, rcond=None)
    #print(rank)
    #print(residuals)
    b_hat = np.dot(A, x)
    if np.allclose(b_hat, b):
        return True
    else:
        return False


def vec_inv(x):
    """
    Moore-Penrose inverse of a vector
    """
    return x.H / x.H.dot(x)
