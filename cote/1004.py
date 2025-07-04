T = int(input())

for _ in range(T):
    x1, y1, x2, y2 = map(int, input().split())
    n = int(input())
    p_list = []
    ans = 0
    for i in range(n):
        info = list(map(int, input().split()))
        p_list.append(info)

    for i in range(n):
        s_dist = (x1 - p_list[i][0])**2 + (y1 - p_list[i][1])**2
        d_dist = (x2 - p_list[i][0])**2 + (y2 - p_list[i][1])**2

        if(s_dist <= p_list[i][2]**2 and d_dist <= p_list[i][2]**2):
            continue
        elif(s_dist <= p_list[i][2]**2 or d_dist <= p_list[i][2]**2):
            ans+=1
    
    print(ans)
    