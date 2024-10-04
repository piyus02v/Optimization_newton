#include <iostream>
#include <unordered_map>
#include <vector>

using namespace std;

vector<int> max_subsequence_with_sum_mod_k(vector<int>& arr, int k) {
    unordered_map<int, int> mod_hash; // Hash to store the modulus of prefix sums

    mod_hash[0] = -1; // Initialize the hash with modulus 0 at index -1
    int max_length = 0;
    int max_start_index = -1;

    int prefix_sum = 0;
    for (int i = 0; i < arr.size(); ++i) {
        prefix_sum = (prefix_sum + arr[i]) % k; // Calculate prefix sum modulo k

        if (mod_hash.find(prefix_sum) != mod_hash.end()) { // If prefix_sum modulo k is already in the hash
            int length = i - mod_hash[prefix_sum];
            if (length > max_length) { // Update max_length if current length is greater
                max_length = length;
                max_start_index = mod_hash[prefix_sum] + 1;
            }
        } else {
            mod_hash[prefix_sum] = i;
        }
    }

    vector<int> max_subsequence;
    if (max_start_index != -1) {
        max_subsequence = vector<int>(arr.begin() + max_start_index, arr.begin() + max_start_index + max_length); // Extract max subsequence
    }
    return max_subsequence;
}

int main() {
    vector<int> arr = {1,2,3,4};
    int k = 3;
    vector<int> result = max_subsequence_with_sum_mod_k(arr, k);
    cout << "Maximum subsequence:";
    for (int num : result) {
        cout << " " << num;
    }
    cout << endl;
    return 0;
}
