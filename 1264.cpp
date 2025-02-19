#include <iostream>
#include <string>
#include <algorithm>
using namespace std;

int findVowel(const string &inp){
    int len = inp.size();
    int cnt = 0;
    for(int i = 0; i < len; i++){
        if(inp[i] == 'a' || inp[i] == 'e' || inp[i] == 'i' || inp[i] == 'o' || inp[i] == 'u')
            cnt++;
        else if(inp[i] == 'A' || inp[i] == 'E' || inp[i] == 'I' || inp[i] == 'O' || inp[i] == 'U')
            cnt++;
    }
    
    return cnt;
}

int main(){
    string inp;
    
    while(1){
        getline(cin, inp);
        if(inp == "#") break;
        cout << findVowel(inp) << '\n';
    }

    return 0;
}