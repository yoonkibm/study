#include <iostream>
using namespace std;

int calculateLeaf(int a, int *arr){
    int len = 2 * a;
    int leaf = 1;

    for(int i = 0; i < len; i += 2){
        leaf = leaf*arr[i];
        leaf = leaf-arr[i+1];
    }

    return leaf;
}

int main(){
    int a;
    int arr[40];
    int len;

    while(1){
        cin >> a;
        if(a == 0) break;
        len = 2*a;
        for(int i = 0; i < len; i++){
            cin >> arr[i];
        }

        cout << calculateLeaf(a, arr) << '\n';
    }

    return 0;
}