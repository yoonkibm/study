#include <iostream>
#include <algorithm>
using namespace std;

int calculateError(int p, int a, int inp){
    int error;
    int number = p * a;

    error = inp - number;

    return error;
}

int main(){
    int p, area;
    int arr[5];

    cin >> p >> area;

    for(int i = 0; i < 5; i++){
        cin >> arr[i];
    }

    for(int i = 0; i < 5; i++){
        cout << calculateError(p, area, arr[i]) << " ";
    }

    cout << "\n";

    return 0;
}