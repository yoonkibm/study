#include <bits/stdc++.h>
using namespace std;

/*
 * 문자열로 표현된 두 수의 절댓값을 더한다 (부호는 따로 처리).
 * 예: "123" + "999" -> "1122"
 */
string addAbsolute(const string &a, const string &b) {
    // a, b 는 모두 0으로 시작하지 않는다고 가정 (미리 정규화된 상태)
    // 자릿수 차이가 있을 수 있으므로 뒤에서부터 더해 나간다.
    int carry = 0;
    int i = a.size() - 1, j = b.size() - 1;
    string result;
    while(i >= 0 || j >= 0 || carry) {
        int sum = carry;
        if(i >= 0) sum += (a[i--] - '0');
        if(j >= 0) sum += (b[j--] - '0');
        carry = sum / 10;
        result.push_back((sum % 10) + '0');
    }
    reverse(result.begin(), result.end());
    return result;
}

/*
 * 문자열로 표현된 두 수의 절댓값 중 a >= b 라고 가정하고,
 * a - b(절댓값)를 계산한다. (a, b 모두 non-negative, a >= b 상태)
 * 예: "1000" - "1" -> "999"
 */
string subAbsolute(const string &a, const string &b) {
    // a의 자릿수가 b의 자릿수보다 길거나 같고, a >= b 라고 가정
    int i = a.size() - 1, j = b.size() - 1;
    int borrow = 0;
    string result;
    while(i >= 0 || j >= 0) {
        int x = (i >= 0 ? a[i--] - '0' : 0) - borrow;
        int y = (j >= 0 ? b[j--] - '0' : 0);
        int diff = x - y;
        if(diff < 0) {
            diff += 10;
            borrow = 1;
        } else {
            borrow = 0;
        }
        result.push_back(diff + '0');
    }
    // 앞쪽에 생긴 불필요한 0을 제거
    while(result.size() > 1 && result.back() == '0') {
        result.pop_back();
    }
    reverse(result.begin(), result.end());
    return result;
}

/*
 * 문자열로 표현된 두 수의 절댓값을 곱한다 (부호는 따로 처리).
 * 예: "123" × "45" -> "5535"
 */
string mulAbsolute(const string &a, const string &b) {
    // a, b 가 "0"이면 바로 "0" 리턴
    if(a == "0" || b == "0") return "0";

    vector<int> v(a.size() + b.size());
    // a, b 뒤에서부터 곱셈 수행
    for(int i = a.size() - 1; i >= 0; i--) {
        for(int j = b.size() - 1; j >= 0; j--) {
            int mul = (a[i] - '0') * (b[j] - '0');
            mul += v[i + j + 1];
            v[i + j + 1] = mul % 10;
            v[i + j] += mul / 10;
        }
    }
    // 벡터에 저장된 결과를 문자열로 변환
    string result;
    for(int i = 0; i < (int)v.size(); i++) {
        // 가장 앞의 불필요한 0은 건너뛰되, 전체가 0인 경우는 제외
        if(!(result.empty() && v[i] == 0)) {
            result.push_back(v[i] + '0');
        }
    }
    if(result.empty()) return "0"; // 모두 0이었던 경우
    return result;
}

/*
 * 문자열 형태의 큰 정수 두 개의 크기를 비교 (절댓값 기준)
 * 리턴값: |a| < |b| 이면 -1, |a| == |b| 이면 0, |a| > |b| 이면 1
 * (a, b는 부호 없는 상태로 받는다고 가정)
 */
int compareAbsolute(const string &a, const string &b) {
    if(a.size() < b.size()) return -1;
    if(a.size() > b.size()) return 1;
    return a.compare(b) == 0 ? 0 : (a.compare(b) < 0 ? -1 : 1);
}

/*
 * 문자열 앞의 불필요한 0을 제거. (단, "0" 자체는 유지)
 * 예: "000123" -> "123"
 */
string removeLeadingZeros(const string &s) {
    // s가 음수가 아닌 문자열이라 가정
    int idx = 0;
    while(idx + 1 < (int)s.size() && s[idx] == '0') {
        idx++;
    }
    return s.substr(idx);
}

/*
 * 문자열 형태의 정수를 (+/-) 형태로 읽어서, 
 * sign(부호)과 absVal(절댓값)로 분리
 */
pair<int, string> parseInteger(const string &str) {
    // sign: +1 또는 -1
    // absVal: 앞자리부터 0이 아닌 부분까지 잘라내고 저장
    if(str[0] == '-') {
        return {-1, removeLeadingZeros(str.substr(1))};
    } else if(str[0] == '+') {
        return {+1, removeLeadingZeros(str.substr(1))};
    } else {
        return {+1, removeLeadingZeros(str)};
    }
}

/*
 * 결과가 0인지 확인하고, 0이 아니면 맨 앞이 '0'으로 시작하지 않도록 정규화
 */
string normalizeResult(const string &res, int sign) {
    // 0인 경우
    if(res == "0") {
        return "0";
    }
    // 0이 아닌데 sign이 음수이면 앞에 '-' 붙이기
    if(sign < 0) {
        return "-" + res;
    }
    // 양수이면 그대로
    return res;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string A, B;
    cin >> A >> B;

    // 1) 입력을 (부호, 절댓값) 형태로 파싱
    auto [signA, absA] = parseInteger(A);
    auto [signB, absB] = parseInteger(B);

    // -----------------------
    // A + B 계산
    // -----------------------
    int signAdd;      // (A+B)의 부호
    string absAdd;    // (A+B)의 절댓값

    if(absA == "0" && absB == "0") {
        // 둘 다 0이면 결과는 0
        signAdd = +1;
        absAdd = "0";
    } else if(signA == signB) {
        // 부호가 같으면 절댓값을 더하고, 부호는 그대로
        signAdd = signA;
        absAdd = addAbsolute(absA, absB);
    } else {
        // 부호가 다르면 절댓값이 큰 쪽에서 작은 쪽을 뺀다.
        int cmp = compareAbsolute(absA, absB);
        if(cmp == 0) {
            // 절댓값이 같으면 결과 0
            signAdd = +1;
            absAdd = "0";
        } else if(cmp > 0) {
            // |A| > |B| 이므로 결과의 부호는 A의 부호
            signAdd = signA;
            absAdd = subAbsolute(absA, absB);
        } else {
            // |A| < |B| 이므로 결과의 부호는 B의 부호
            signAdd = signB;
            absAdd = subAbsolute(absB, absA);
        }
    }

    // -----------------------
    // A - B 계산
    // -----------------------
    int signSub;      // (A-B)의 부호
    string absSub;    // (A-B)의 절댓값

    // A - B = A + (-B)
    // 즉, B의 부호를 반대로 해서 A와 더하기
    int signBInv = -signB;
    if(absA == "0" && absB == "0") {
        signSub = +1;
        absSub = "0";
    } else if(signA == signBInv) {
        // 부호가 같으니 절댓값 더하기
        signSub = signA;
        absSub = addAbsolute(absA, absB);
    } else {
        // 부호가 다르면 절댓값 큰쪽 - 작은쪽
        int cmp = compareAbsolute(absA, absB);
        if(cmp == 0) {
            signSub = +1;
            absSub = "0";
        } else if(cmp > 0) {
            signSub = signA;
            absSub = subAbsolute(absA, absB);
        } else {
            signSub = signBInv;
            absSub = subAbsolute(absB, absA);
        }
    }

    // -----------------------
    // A * B 계산
    // -----------------------
    // 부호: signA * signB (0 인 경우 체크)
    int signMul;
    if(absA == "0" || absB == "0") {
        signMul = +1;
    } else {
        signMul = (signA == signB) ? +1 : -1;
    }
    string absMul = mulAbsolute(absA, absB);

    // -----------------------
    // 출력
    // 0이 아닌 경우에는 앞에 0이 오지 않도록 하고, 0이면 "0"만 출력
    // -----------------------
    cout << normalizeResult(absAdd, signAdd) << "\n";
    cout << normalizeResult(absSub, signSub) << "\n";
    cout << normalizeResult(absMul, signMul) << "\n";

    return 0;
}
