#include <iostream>
#include <algorithm>
using namespace std;

int main(){
    int add, sub;

    cin >> add >> sub;

    if((add+sub)%2 == 1 || add-sub < 0){
        cout << -1 << "\n";
    }
    else
        cout << (add+sub)/2 << " " << (add-sub)/2 << "\n";
    return 0;
}