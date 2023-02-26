import numpy as np
from scipy.stats.distributions import chi2

def drawellipse(x,a,b,colour):
    Npoints = 100
    vec = np.linspace(0,2*np.pi,Npoints)
    p = np.zeros((2,Npoints))
    p[0,:] = a*np.cos(vec)
    p[1,:] = b*np.sin(vec)
    
    # Translate and rotate
    xo = np.squeeze(x[0])
    # print("xo",xo)
    yo = np.squeeze(x[1]) 
    angle = np.squeeze(x[2])

    # print(f"angle:{angle}")
    # Earlier R was np.matrix, it did not plot. 
    R = np.zeros((2,2))
    R  = np.array([[np.cos(angle), -np.sin(angle)], 
                   [np.sin(angle), np.cos(angle)]])

    T = np.zeros((2,1))
    T = np.array([[xo], [yo]], dtype='float')
    # print(f"T.shape:", T.shape)
    # print("np.size(vec):", np.size(vec))
    T  = T@np.ones((1,np.size(vec)))
    # print(np.shape(p))
    # print(np.shape(R))
    # print(np.shape(T))
    # print(T)
    p = R@p + T
    # print(p.shape)
    return p
    # plt.plot(p[0,:], p[1,:], colour)

def drawprobellipse(x,C,alpha,colour):
    sxx = C[0,0]; syy = C[1,1]; sxy = C[0,1]
    a = np.sqrt(0.5*(sxx+syy+np.sqrt((sxx-syy)**2+4*sxy**2)))   # always greater
    # c = 0.5*(sxx+syy-np.sqrt((sxx-syy)**2+4*sxy**2))
    # print("c in ellipse:",c)
    b = np.sqrt(0.5*(sxx+syy-np.sqrt((sxx-syy)**2+4*(sxy**2))))   # always smaller

    # Remove imaginary parts in case of neg. definite C
    if not np.isreal(a): 
        a = np.real(a)
    if not np.isreal(b):
        b = np.real(b)
    
    # Scaling in order to reflect specified probability
    a = a*np.sqrt(chi2.ppf(alpha,df=2))
    b = b*np.sqrt(chi2.ppf(alpha,df=2))

    # Look where the greater half axis belongs to
    if sxx < syy:
        swap = a
        a = b
        b = swap
    
    angle = 0

    # Calculate inclination (numerically stable)
    if sxx != syy:
        angle = 0.5*np.arctan(2*sxy/(sxx-syy));	
    elif sxy == 0:
        angle = 0;     # angle doesn't matter   '
    elif sxy > 0:
        angle =  np.pi/4
    elif sxy < 0:
        angle = -np.pi/4;
    
    # print(f"x.shape:{x.shape}")
    # x[2] = angle; # It has been corrected ad-hoc in the function call statement
    # print(f"a:{a}, b:{b}")
    # print("x:",x)
    return drawellipse([x[0],x[1],angle],a,b,colour)