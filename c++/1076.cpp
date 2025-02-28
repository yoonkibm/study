#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;

vector<string> resistance = {"black", "brown", "red", "orange", "yellow", 
    "green", "blue", "violet", "grey", "white"};

int findIndex(string word){
    auto it = find(resistance.begin(), resistance.end(), word);
    if(it != resistance.end()){
        return distance(resistance.begin(), it);
    }

    return -1;
}

long long calculateResistance(string *arr){
    long long answer;
    int first_index = findIndex(arr[0]);
    int second_index = findIndex(arr[1]);
    int third_index = findIndex(arr[2]);

    answer = (first_index*10 + second_index)*pow(10, third_index);

    return answer;
}

int main(){
    string arr[3];

    cin >> arr[0];
    cin >> arr[1];
    cin >> arr[2];

    long long ans = calculateResistance(arr);

    cout << ans << '\n';
    return 0;
}