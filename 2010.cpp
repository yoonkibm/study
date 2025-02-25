#include <iostream>

using namespace std;

int findConnection(int n){
    int ans = 1;
    int multi_tab;

    for(int i = 0; i < n; i++){
        cin >> multi_tab;

        ans = ans + multi_tab - 1;
    }

    return ans;
}

int main(){
    int n;

    cin >> n;

    cout << findConnection(n) << '\n';

    return 0;
}