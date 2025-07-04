from collections import defaultdict
n = 4

def stair_climbing(n):
    dp = defaultdict(int)
    dp[1] = 1
    dp[2] = 2
    if n <= 2:
        return n
    
    else:
        for i in range(3, n+1):
            dp[i] = dp[i-1] + dp[i - 2]

    return dp[n]

print(stair_climbing(n))