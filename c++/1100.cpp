#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int findWhite(vector<string> vec){
    int ans = 0;
    for(int i = 0; i < 8; i++){
        for(int j = 0; j < 8; j++){
            if(vec[i][j] == 'F'){
                if((i+j)%2 == 0) ans++;
                else continue;
            }
        }
    }

    return ans;
}

int main(){
    vector<string> arr(8);

    for(int i = 0; i < 8; i++){
        cin >> arr[i];
    }

    cout << findWhite(arr) << '\n';
}