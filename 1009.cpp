#include <iostream>
using namespace std;

int findComputer(int a, int b){
    int ans;
    for(int i = 0; i < b; i++){
        if(i == 0) ans = a%10;
        else ans = ans*a%10;
    }
    
    if(ans == 0) ans = 10;
    return ans;
}

int main(){
    int t, a, b;

    cin >> t;
    for(int i = 0; i < t; i++){
        cin >> a >> b;
        cout << findComputer(a, b) << '\n';
    }

    return 0;
}