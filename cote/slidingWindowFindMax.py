from collections import deque

nums = [1, 3, -1, -3, -1, 3, 6, 7]
k = 3

def sliding_window_find_max(nums, k):
    q = deque()
    res = []

    for i, v in enumerate(nums):
        while q and nums[q[-1]] <= v:
            q.pop()
        
        while q and q[0] <= i - k:
            q.popleft()
            print(q)

        q.append(i)

        if i >= k - 1:
            res.append(nums[q[0]])

    return res


answer = sliding_window_find_max(nums, k)

print(answer)

