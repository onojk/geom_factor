import math
import matplotlib.pyplot as plt

def first_forbidden_divisor_counted(n: int):
    if n < 2:
        return (1, 0)

    checks = 0
    limit = int(math.isqrt(n))
    for d in range(2, limit + 1):
        checks += 1
        if n % d == 0:
            return (d, checks)   # composite
    return (None, checks)        # prime

def collect(primes_to_collect=2000):
    n = 2
    last_prime = None

    mods_since = 0
    tested_since = 0

    primes = []
    avg_tests = []

    found = 0
    while found < primes_to_collect:
        d, checks = first_forbidden_divisor_counted(n)

        tested_since += 1
        mods_since += checks

        if d is None:
            avg = mods_since / tested_since
            primes.append(n)
            avg_tests.append(avg)

            last_prime = n
            mods_since = 0
            tested_since = 0
            found += 1

        n += 1

    return primes, avg_tests

def main():
    primes, avg_tests = collect(primes_to_collect=3000)

    plt.figure(figsize=(12, 6))
    plt.plot(primes, avg_tests, ".", markersize=3)
    plt.title("Average mod tests per number between primes")
    plt.xlabel("prime p")
    plt.ylabel("avg tests per number")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()   # ← shows graph directly, no files

if __name__ == "__main__":
    main()
