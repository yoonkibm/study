#include <stdio.h>
#include <stdlib.h>

#define MAX_N 25

char arr[MAX_N][MAX_N];
int flag[MAX_N][MAX_N] = {0};
int cnt[313] = {0};

int mov[][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

void findComplex(int r, int c, int flag_num, int n) {
    // 1. 범위 초과 시 탈출
    if (r < 0 || c < 0 || r >= n || c >= n) return;

    // 2. 이미 방문한 경우 탈출
    if (flag[r][c] != 0) return;

    // 3. 현재 위치 방문 처리
    flag[r][c] = flag_num;
    cnt[flag_num]++;

    // 4. 4방향 DFS 탐색
    for (int i = 0; i < 4; i++) {
        int nr = r + mov[i][0];
        int nc = c + mov[i][1];

        // 범위 안이면서, 아직 방문 안 했고, 집('1')이 있는 경우만 재귀
        if (nr >= 0 && nc >= 0 && nr < n && nc < n) {
            if (arr[nr][nc] == '1' && flag[nr][nc] == 0) {
                findComplex(nr, nc, flag_num, n);
            }
        }
    }
}


int main(){
    int N;
    scanf("%d", &N);
    getchar();

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            arr[i][j] = getchar();
        }
        getchar();
    }

    int flag_num = 1;

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            if(arr[i][j] == '1' && flag[i][j] == 0){
                findComplex(i, j, flag_num, N);
                flag_num++;
            }
        }
    }
    
    if(flag_num >= 2){
        printf("%d\n", flag_num-1);

        if(flag_num == 2) printf("%d\n", cnt[1]);
        else{
            for (int i = 1; i < flag_num - 1; i++) {
                for (int j = 1; j < flag_num - i; j++) {
                    if (cnt[j] > cnt[j + 1]) {
                        int temp = cnt[j];
                        cnt[j] = cnt[j + 1];
                        cnt[j + 1] = temp;
                    }
                }
            }

            for(int i = 1; i < flag_num; i++){
                printf("%d\n", cnt[i]);
            }
        }
    }

    else if(flag_num == 1){
        printf("0\n");
    }
    return 0;
}