import numpy as np
import math
import random

# (2*x-5)**4 - (x**2-1)**3 max (-10,0)
# 8+x**3-2*x-2*math.e**x max (-2,1)
# 4*x*math.sin(x) max (0.5,pi)
# 2*(x-3)**2 + math.e**(0.5*x**2) min (-2,3)
# x**2 - 10*math.e**(0.1*x) min (-6,6)
# 20*math.sin(x) - 15*x**2 max (-4,4)


def func(x):
    # return (2*x-5)**4 - (x**2-1)**3
    # return 8+x**3-2*x-2*math.e**x
    # return 4*x*math.sin(x)
    # return 2*(x-3)**2 + math.e**(0.5*x**2)
    # return x**2 - 10*math.e**(0.1*x)
    return 20*math.sin(x) - 15*x**2


def bounding_phase(lb, ub, opt):

    x = random.uniform(lb, ub)
    print(f'Stating val: {x}')
    dx = 0.00001

    x_lb = x-dx
    x_ub = x+dx
    f_x = func(x)
    f_x_lb = func(x_lb)
    f_x_ub = func(x_ub)
    k = 0

    while x_ub<=ub and x_lb>=lb:

        if opt == 'min':
            if f_x<=f_x_lb and f_x>=f_x_ub:
                dx = abs(dx)
            elif f_x>=f_x_lb and f_x<=f_x_ub:
                dx = -abs(dx)
            
        elif opt == 'max':
            if f_x<=f_x_lb and f_x>=f_x_ub:
                dx = -abs(dx)
            elif f_x>=f_x_lb and f_x<=f_x_ub:
                dx = abs(dx)

        k = k+1
        x_n = x + (2**k)*dx
        f_x_n = func(x_n)

        if opt == 'min':
            if f_x_n>f_x:
                if x_n>ub:
                    x_n = ub
                elif x_n<lb:
                    x_n = lb
                return [x, x_n]
        elif opt == 'max':
            if f_x_n<f_x:
                if x_n>ub:
                    x_n = ub
                elif x_n<lb:
                    x_n = lb
                return [x, x_n]
    
        x = x_n
        x_ub = x+abs(dx)
        x_lb = x-abs(dx)
        f_x = f_x_n
        f_x_lb = func(x_lb)
        f_x_ub = func(x_ub)

    return [x, x_n]

def interval_halving(lb, ub, opt):

    eps = 1e-5
    xm = (lb+ub)/2 
    L = ub - lb 
    x_l = lb + L/4  
    x_u = ub - L/4 

    f_u = func(x_u)
    f_l = func(x_l)
    f_m = func(xm)

    while abs(L)>eps:

        if opt == 'min':
            if f_l<f_m:
                ub = xm
                xm = x_l
            elif f_u<f_m:
                lb=xm
                xm = x_u
            else:
                lb = x_l
                ub = x_u

        elif opt == 'max':
            if f_l>f_m:
                ub = xm
                xm = x_l
            elif f_u>f_m:
                lb=xm
                xm = x_u
            else:
                lb = x_l
                ub = x_u    
        
        L = ub - lb 
        x_l = lb + L/4  
        x_u = ub - L/4 
        f_l = func(x_l)
        f_u = func(x_u)
        f_m = func(xm)

    return xm

def main():
    lower_bound = input('Enter Lower Bound: ')
    upper_bound = input('Enter Upper Bound: ')
    opti = input('Enter whether the func need to be min or max: ')
    
    if upper_bound[-1] == 'i':
        
        if len(upper_bound) == 2:
            upper_bound = float(math.pi)
        else:
            upper_bound = float(upper_bound[:-2])*math.pi
    
    if lower_bound[-1] == 'i':
        
        if len(lower_bound) == 2:
            lower_bound = float(math.pi)
        else:
            lower_bound = float(lower_bound[:-2])*math.pi
    
    else:
        upper_bound = float(upper_bound)
        lower_bound = float(lower_bound)


    for i in range(10):
        li = bounding_phase(lower_bound, upper_bound, opti)
        li.sort()
        print(f'Final Range: {li}')
        final_ans = interval_halving(li[0], li[1], opti)
        print(f'final_ans: {final_ans}')

if __name__ == '__main__':
    main()