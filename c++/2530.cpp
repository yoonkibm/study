#include <iostream>
#include <algorithm>
using namespace std;

tuple<int, int, int> calculateTime(int h, int m, int s, int t){
    int carry = 0;
    
    s = s + t;
    carry = s/60;
    s = s%60;

    m = m+carry;
    carry = m/60;
    m = m%60;

    h = h+carry;
    h = h%24;


    return make_tuple(h, m, s);
}

int main(){
    int hour, minute, second, time;

    cin >> hour >> minute >> second;
    cin >> time;

    auto [h, m, s] = calculateTime(hour, minute, second, time);

    cout << h << " " << m << " " << s << '\n';
}