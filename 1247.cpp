#include <iostream>
#include <algorithm>
#include <deque>
using namespace std;

int main(){
    int n;
    int cnt = 0;
    deque<long long> pos;
    deque<long long> neg;
    char buho;
    long long buf;
    bool is_p, is_n;

    while(cnt < 3){
        cnt++;
        is_p = true;
        is_n = true;
        pos.clear();
        neg.clear();
        long long ans = 0;
        cin >> n;
        for(int i = 0; i < n; i++){
            cin >> buf;
            if(buf > 0) pos.push_back(buf);
            else if(buf == 0) continue;
            else neg.push_back(buf);
        }
        
        while(1){
            if(neg.empty() && pos.empty() && ans == 0){
                buho = '0';
                break;
            }
            else if(neg.empty() && ans >= 0){
                buho = '+';
                break;
            }
            else if(pos.empty() && ans =< 0){
                buho = '-';
                break;
            }
            if(is_p && is_n){
                ans = pos[0] + neg[0];
            }
            else if(is_p){
                ans = pos[0] + ans;
            }
            else if(is_n){
                ans = neg[0] + ans;
            }
            if(ans > 0) {
                neg.pop_front();
                is_p = false;
                is_n = true;
            }
            else if(ans < 0) {
                pos.pop_front();
                is_p = true;
                is_n = false;
            }
            else{
                pos.pop_front();
                neg.pop_front();
                is_p = true;
                is_n = true;
            }
        }
        cout << buho << '\n';
    }

    return 0;
}