#include <iostream>
#include <string>
#include <algorithm>
using namespace std;

int calculateMoney(int f, int s, int t){
    if(f == s && s == t){
        return 10000+f*1000;
    }
    else if(f != s && s != t && f != t){
        return 100*max(f, max(s, t));
    }
    else{
        if(f == s) return 1000+f*100;
        if(f == t) return 1000+f*100;
        if(s == t) return 1000+s*100;
    }
}

int main(){
    int first, second, third;

    cin >> first >> second >> third;

    cout << calculateMoney(first, second, third);

    return 0;
}