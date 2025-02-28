#include <iostream>

using namespace std;

int findMax(){
    int in, out;
    int person = 0, max_person = 0;

    // 2460은 i가 9까지 증가하게 하면 됨됨
    for(int i = 0; i < 4; i++){
        cin >> out >> in;
        person = person + in - out;
        if(person > max_person) max_person = person;
    }

    return max_person;
}

int main(){
    cout << findMax() << '\n';

    return 0;
}