#include <iostream>
#include <string>
#include <algorithm>
using namespace std;

string classification(int a, int w){
    if(a > 17 || w >= 80) return "Senior";
    else return "Junior";
}

int main(){
    string name;
    int age, weight;

    while(1){
        cin >> name >> age >> weight;
        if(name == "#") break;
        cout << name << " " << classification(age, weight) << '\n';
    }

    return 0;
}