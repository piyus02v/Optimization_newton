#include <iostream>
#include <cctype>


// Q1
int main() {
    std::string s;
    std::getline(std::cin, s);

    int letters = 0, spaces = 0, numbers = 0, others = 0;

    for (char c : s) {
        if (std::isalpha(c))
            letters++;
        else if (std::isspace(c))
            spaces++;
        else if (std::isdigit(c))
            numbers++;
        else
            others++;
    }

    std::cout << "Letters: " << letters << std::endl;
    std::cout << "Spaces: " << spaces << std::endl;
    std::cout << "Numbers: " << numbers << std::endl;
    std::cout << "Other characters: " << others << std::endl;

    return 0;
}



// Q2 
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    cin>>n;
    int arr[n];
    for(int i=0;i<n;i++) {
        cin>>arr[i];
    }
    int count = 0;
    int prev = -1;

    for (int num : arr) {
        if (num != 0 && num != prev) {
            count++;
            prev = num;
        }
    }

    cout << count << endl;

    prev = -1;
    for (int num : arr) {
        if (num != 0 && num != prev) {
            cout << num << " ";
            prev = num;
        }
    }

    return 0;
}