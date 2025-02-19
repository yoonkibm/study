#include <iostream>
#include <string>

using namespace std;

int calculateLength(const string &inp){
    int ans = 2;
    auto len = inp.size();
    ans = ans + len - 1;
    for(int i = 0; i < len; i++){
        if(inp[i] == '1') ans += 2;
        else if(inp[i] == '0') ans += 4;
        else ans += 3;
    }

    return ans;
}

int main(){
    string inp;

    while(1){
        cin >> inp;
        if(inp == "0") break;
        cout << calculateLength(inp) << '\n';

    }

    return 0;
}