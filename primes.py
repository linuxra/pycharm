def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def is_palindrome(n):
    s = str(n)
    return s == s[::-1]


def find_prime_palindromes(n):
    for i in range(n + 1):
        if is_prime(i) and is_palindrome(i):
            yield i


if __name__ == "__main__":
    for prime_palindrome in find_prime_palindromes(1000000):
        print(prime_palindrome)