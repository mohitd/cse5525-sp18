import numpy as np

def sumdist(x1,x2,distfunc):
    return np.sum(distfunc(x1, x2))

def absdiff(x, y):
    return abs(x-y)

def euclidean(x, y):
    return np.linalg.norm(x-y)

def dtw(x, y, dist):
    points = []
    gx = 0
    gy = 0
    m = len(x)
    n = len(y)
    D = np.zeros((m, n))
    B = np.zeros((m, n, 2))
    for i in range(m):
        for j in range(n):
            D[i, j] = min([D[i-1, j], D[i-1, j-1], D[i,j-1]]) + dist(x[i], y[j])
            idx = np.argmin([D[i-1, j], D[i-1, j-1], D[i,j-1]])
            if idx == 0:
                B[i, j, 0] = -1
                gx+=1
            elif idx == 1:
                B[i, j, 0] = -1
                B[i, j, 1] = -1
                gx+=1
                gy+=1
            else:
                B[i, j, 1] = -1
                gy+=1
            points.append([gx, gy])
    return D[-1, -1], B, points

if __name__ == '__main__':
    s, B, p = dtw(np.array([1,2,4,2,3]),np.array([1,1,2,2,3]),absdiff)
    print(s)

    import matplotlib
    import matplotlib.pyplot as plt
    plt.quiver(B[:,:,0],B[:,:,1])
    plt.show()

    x = [i[0] for i in p]
    y = [i[1] for i in p]
    
    plt.plot(x, y)
    plt.show()