#include <stdio.h>

void fillMat(int n, int arr[][15]){
    int mov[][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    int dir = 0;

    int inp = 1;
    int cnt = 1;
    int id[2] = {0, 0};
    int flag[15][15] = {0};

    while(cnt <= n*n){
        if(cnt % 2 == 1){
            arr[id[0]][id[1]] = inp;
            inp++;
        }
        else arr[id[0]][id[1]] = 0;

        flag[id[0]][id[1]] = 1;

        int next_i = id[0] + mov[dir][0];
        int next_j = id[1] + mov[dir][1];

        if(next_i < 0 || next_i >= n || next_j < 0 || next_j >= n || flag[next_i][next_j] == 1){
            dir = (dir + 1) % 4;
            next_i = id[0] + mov[dir][0];
            next_j = id[1] + mov[dir][1];   
        }
        id[0] = next_i;
        id[1] = next_j;
        cnt++;
    }
}

int main(){

    int N;
    printf("입력:");
    scanf("%d", &N);

    int ans[15][15] = {0};

    fillMat(N, ans);

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            if(N >= 10) printf("%03d ", ans[i][j]);
            else printf("%02d ", ans[i][j]);
        }
        printf("\n");
    }

    return 0;
}