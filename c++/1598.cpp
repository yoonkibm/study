#include <iostream>
#include <cmath>
using namespace std;

int main(){
    int a, b;
    int x[2];
    int y[2];

    cin >> a >> b;

    x[0] = a/4;
    x[1] = b/4;
    if(a%4 == 0) {
        y[0] = 4;
        x[0] -= 1;
    }
    else y[0] = a%4;
    if(b%4 == 0) {
        y[1] = 4;
        x[1] -= 1;
    }
    else y[1] = b%4;

    cout << abs(x[0]-x[1])+abs(y[0]-y[1]) << '\n';

    return 0;
}