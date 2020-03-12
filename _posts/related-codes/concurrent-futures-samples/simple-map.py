from concurrent.futures import ThreadPoolExecutor

def my_pow(base, power):
    '''
    say we want threads to run a stupid power function
    '''
    result = 1
    for i in range(power):
        result = result * base
    return result

base_list = [1, 2, 3, 4, 5, 6]
power_list = [2, 3, 4, 4, 3, 2]
with ThreadPoolExecutor(max_workers=5) as executor:
    result_list = executor.map(my_pow, base_list, power_list)
print(list(result_list))