import numpy as np

def thomas_method_block(As:np.array, Bs:np.array, Cs:np.array, Ds:np.array):
    # Forward process
    gama = np.linalg.inv(Bs[0])
    b_star = [gama.dot(Bs[0])]
    c_star = [gama.dot(Cs[0])]
    r_star = [gama.dot(Ds[0])]
    for i in range(1,Bs.shape[0]):
        #b_star.append(np.linalg.inv( Bs[i] - As[i].dot(c_star[i-1])))
        #c_star.append(b_star[i].dot( Cs[i]                          ))
        #r_star.append(b_star[i].dot( Ds[i] - As[i].dot(r_star[i-1]) ))
        gama = np.linalg.inv(Bs[i]-As[i].dot(c_star[i-1]))
        b_star.append(gama.dot(Bs[i]-As[i].dot(c_star[i-1])))
        c_star.append(gama.dot(Cs[i]))
        r_star.append(gama.dot(Ds[i]-As[i].dot(r_star[i-1])))

    # Backward process
    response = np.array([r_star[-1]]).reshape((1,-1))
    for i in range(Bs.shape[0]-2, -1, -1):
        response = np.insert(response, 0, r_star[i]-c_star[i].dot(response[0]), axis=0)

    return response

def thomas_method_block2(As, Bs, Cs, Ds):
    """
    [
    [Bs[0],Cs[0], zeros       ]
    [As[1],Bs[1], zeros       ]
    [zeros,As[2], Bs[2], Cs[2]]
    [          ...            ]
    [          ..             ]
    [          .  .           ]
    [          .    .         ]
    [      zeros, As[2], BS[2]]
    ]
    """

    # Forward process
    b_star = [np.linalg.inv(Bs[0])]
    c_star = [b_star[0].dot(Cs[0])]
    r_star = [b_star[0].dot(Ds[0])]
    for i in range(1,Bs.shape[0]):
        b_star.append(np.linalg.inv(As[i]).dot(Bs[i]) - c_star[-1])
        c_star.append(np.linalg.inv(b_star[i]).dot(np.linalg.inv(As[i]).dot(Cs[i])))
        r_star.append(np.linalg.inv(b_star[i]).dot(np.linalg.inv(As[i]).dot(Ds[i]) - r_star[-1]))

    # Backward process
    response = np.array([r_star[-1]]).reshape((1,-1))
    for i in range(Bs.shape[0]-2, -1, -1):
        response = np.insert(response, 0, r_star[i]-c_star[i].dot(response[-1]), axis=0)

    return response


def thomas_method(A, D):
    b = np.diag(A)

    a = np.insert(np.diag(A, k=-1), 0, 0.0)

    c = np.append(np.diag(A, k=1), 0.0)
    beta = [b[0]]
    for i in range(1,a.shape[0]):
        beta.append(b[i]-a[i]*c[i-1]/beta[i-1])

    gama = [D[0]/b[0]]
    for i in range(1, a.shape[0]):
        gama.append((D[i] - a[i]*gama[i-1]) / beta[i])

    y = [gama[-1]]
    for i in range(a.shape[0]-2, -1, -1):
        y.insert(0, gama[i]-c[i]*y[0]/beta[i])
    return y

