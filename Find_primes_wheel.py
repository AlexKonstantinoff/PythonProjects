import numpy as np

def is_prime(n, primes, wheel, wheelsize):
    for i in primes:
        if n == i:
            return True
        if not(n % i):
            return False

        while True:
            for j in wheel:
                if j**2>n:
                    return 1
                if not (n % j):
                    return 0

                wheel.append(j + wheelsize)          

def Wheel(f_max):
    primes = [2, 3, 5, 7]
    wheelsize = np.prod(primes)
    
    wheel = []
    for i in range(2, wheelsize + 1):
        for j in primes:
            if not(i % j):
                continue
        wheel.append(i)

    for i in range(8, f_max):
        if is_prime(i, primes, wheel, wheelsize):
            primes.append(i)
    
    return primes

print(Wheel(120))
