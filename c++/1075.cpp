#include <iostream>
using namespace std;

char* findAnswer(int n, int f){
    n = n/100*100;
    char* ans = new char[3];
    
    for(int i = 0; i < 100; i++){
        if(n%f == 0) break;
        else n++;
    }

    n = n%100;

    ans[0] = n/10 + 48;
    ans[1] = n%10 + 48;
    ans[2] = '\0';

    return ans;
}

int main(){
    int n, f;

    cin >> n;
    cin >> f;

    char* answer = findAnswer(n, f);

    cout << answer << '\n';

    delete[] answer;
    return 0;
}