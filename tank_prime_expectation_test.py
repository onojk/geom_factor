#!/usr/bin/env python3
import math


def sieve(n: int):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(math.isqrt(n)) + 1):
        if is_prime[p]:
            for m in range(p * p, n + 1, p):
                is_prime[m] = False
    return is_prime


def main():
    N = 1000
    is_prime = sieve(N)

    actual = 0

    print("n\tactual_primes\texpected_primes")
    print("-" * 40)

    for n in range(2, N + 1):
        if is_prime[n]:
            actual += 1

        expected = n / math.log(n)
        print(f"{n}\t{actual}\t\t{expected:.2f}")


if __name__ == "__main__":
    main()
