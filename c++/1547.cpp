#include <iostream>
#include <vector>

using namespace std;

int main(){
    int m;
    int x, y;
    int a = 1, b = 2, c = 3, temp = 0;

    cin >> m;
    for(int i = 0; i < m; i++){
        cin >> x >> y;
        if((x == a && y == b) || (x == b && y == a)){
            temp = a;
            a = b;
            b = temp;
        }
        else if((x == b && y == c) || (x == c && y == b)){
            temp = b;
            b = c;
            c = temp;
        }
        else if((x == a && y == c) || (x == c && y == a)){
            temp = a;
            a = c;
            c = temp;
        }
    }

    cout << a << '\n';

    return 0;
}