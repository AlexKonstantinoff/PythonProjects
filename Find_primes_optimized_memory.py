def find_primes_optimized_memory(f_num):
    primes = [2]

    for i in range(3, f_num, 2):
        flag = True
        for j in primes:
            if j**2 > i:
                break
            if i % j == 0:
                flag = False
                break
        if flag:
            primes.append(i)

    return primes
    
f = int(input('Введите конечное число: '))
res = find_primes_optimized_memory(f)
print(res)