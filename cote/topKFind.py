from collections import defaultdict
import heapq

stream = ["apple", "banana", "apple", "orange", "banana", "apple", "grape"]
k = 4

def top_k_find(stream, k):
    freq = defaultdict(int)
    heap = []

    for item in stream:
        freq[item] += 1
    
    for item, count in freq.items():
        heapq.heappush(heap, (count, item))
        if len(heap) > k:
            heapq.heappop(heap)



    return [id for (cnt, id) in sorted(heap, key = lambda x: -x[0])]

print(top_k_find(stream, k))
