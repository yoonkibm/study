#include <stdio.h>
#include <stdlib.h>

#define MAX_N 25

char arr[MAX_N][MAX_N];
int flag[MAX_N][MAX_N] = {0};
int cnt[313] = {0};

int mov[][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

void findComplex(int r, int c, int flag_num, int n){
    if (flag[r][c] != 0) return;
    
    flag[r][c] = flag_num;
    cnt[flag_num]++;
    for(int i = 0; i < 4; i++){
        if(r + mov[i][0] >= 0 && c + mov[i][1] >= 0 && r + mov[i][0] < n && c + mov[i][1] < n){
            if(arr[r+mov[i][0]][c+mov[i][1]] == '1'){
                findComplex(r+mov[i][0], c+mov[i][1], flag_num, n);
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