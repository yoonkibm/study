#include <iostream>
#include <string>
#include <algorithm>

using namespace std;

int compareStrings(const string &a, const string &b){
    if(a.size() > b.size()) return 1;
    if(a.size() < b.size()) return -1;

    for(int i = 0; i < a.size(); i++){
        if(a[i] > b[i]) return 1;
        if(a[i] < b[i]) return -1;
    }

    return 0;
}

string subtractStrings(const string &a, const string &b){
    string result;

    int carry = 0;
    int i = a.size() - 1;
    int j = b.size() - 1;
    while(i >= 0 || j >= 0 || carry){
        int x = (i >= 0 ? a[i] - '0' : 0);
        int y = (j >= 0 ? b[j] - '0' : 0);
        x = x - y - carry;
        carry = 0;
        if (x < 0) {
            x += 10;
            carry = 1;
        }
        result.push_back((char)(x + '0'));
        i--; j--;
    }

    while (result.size() > 1 && result.back() == '0')
        result.pop_back();
    reverse(result.begin(), result.end());
    return result;
}

string multiplyString(const string &m, int x){
    if(x == 0 || m == "0") return "0";

    int carry = 0;
    string result;

    for (int i = m.size() - 1; i >= 0; i--) {
        int prod = (m[i] - '0') * x + carry;
        carry = prod / 10;
        prod = prod % 10;
        result.push_back((char)(prod + '0'));
    }
    if (carry) result.push_back((char)(carry + '0'));
    reverse(result.begin(), result.end());
    return result;
}

pair<string, string> longDivision(const string &n, const string &m) {
    // n < m 이면 몫은 0, 나머지는 n
    if (compareStrings(n, m) < 0) {
        return make_pair("0", n);
    }
    // n == m 이면 몫은 1, 나머지는 0
    if (compareStrings(n, m) == 0) {
        return make_pair("1", "0");
    }

    string result;     // 몫
    string cur;        // 현재 나머지를 저장할 문자열

    // n의 각 자리 순회
    for (int i = 0; i < (int)n.size(); i++) {
        // 현재 자릿수를 cur에 추가
        cur.push_back(n[i]);
        // 불필요한 앞자리 0 제거
        while (cur.size() > 1 && cur[0] == '0') {
            cur.erase(cur.begin());
        }
        
        // cur와 m을 비교해서, 0~9 중 얼마나 나누어 떨어지는지 찾기
        // (m이 큰 수이므로, 단순히 int 변환 불가. compareStrings 사용)
        int quotientDigit = 0;
        // 선형으로 1~9까지 테스트 (큰 수이지만 곱하는 대상은 최대 9이므로 OK)
        for (int x = 1; x <= 9; x++) {
            string tmp = multiplyString(m, x);
            // tmp <= cur 이면 후보
            if (compareStrings(tmp, cur) <= 0) {
                quotientDigit = x;
            } else {
                break;
            }
        }
        
        // 구한 몫 자리만큼 곱한 값을 cur에서 빼서 새로운 cur를 만든다
        if (quotientDigit > 0) {
            string toSubtract = multiplyString(m, quotientDigit);
            cur = subtractStrings(cur, toSubtract);
        }
        
        // 몫 자릿수 추가
        result.push_back((char)('0' + quotientDigit));
    }

    // 몫(result) 앞의 불필요한 0 제거
    while (result.size() > 1 && result[0] == '0') {
        result.erase(result.begin());
    }
    // 남아있는 cur가 최종 나머지인데, 0 제거(단 '0' 자체면 남겨야 함)
    if (cur.empty()) {
        cur = "0";
    } else {
        // 필요할 경우만 0 제거
        while (cur.size() > 1 && cur[0] == '0') {
            cur.erase(cur.begin());
        }
    }
    
    return make_pair(result, cur);
}

int main(){
    string n, m;
    cin >> n >> m;

    auto [quotient, remainder] = longDivision(n, m);

    cout << quotient << "\n" << remainder << "\n";

    return 0;
}