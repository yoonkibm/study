from collections import deque

graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['B', 'C', 'E'],
    'E': ['D', 'F'],
    'F': ['E']
}

def graph_bfs(graph, s, e):
    parent = {s: None}
    q = deque([s])
    visited = [s]

    while q:
        u = q.popleft()
        if u == e:
            break

        for v in graph[u]:
            if v not in visited:
                visited.append(v)
                parent[v] = u
                q.append(v)

    if e not in parent:
        return None
    
    path = []
    cur = e

    while cur is not None:
        path.append(cur)
        cur = parent[cur]

    return path[::-1]

print(graph_bfs(graph, 'A', 'F'))


