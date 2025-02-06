#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>

// LCG function
std::vector<int> lcg(int seed, int a = 1664525, int c = 1013904223, int m = 1 << 32) {
    std::vector<int> values;
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, m - 1);
    for (int i = 0; i < 10000; ++i) {
        values.push_back(dis(gen));
    }
    return values;
}

// Maximum subarray sum function
int max_subarray_sum(const std::vector<int>& random_numbers) {
    int max_sum = INT_MIN;
    for (int i = 0; i < random_numbers.size(); ++i) {
        int current_sum = 0;
        for (int j = i; j < random_numbers.size(); ++j) {
            current_sum += random_numbers[j];
            if (current_sum > max_sum) {
                max_sum = current_sum;
            }
        }
    }
    return max_sum;
}

// Total maximum subarray sum function
int total_max_subarray_sum(int n, int initial_seed, int min_val, int max_val) {
    int total_sum = 0;
    std::mt19937 gen(initial_seed);
    std::uniform_int_distribution<> dis(min_val, max_val);
    for (int i = 0; i < 20; ++i) {
        int seed = dis(gen);
        std::vector<int> random_numbers = lcg(seed);
        total_sum += max_subarray_sum(random_numbers);
    }
    return total_sum;
}

int main() {
    // Parameters
    int n = 10000;         // Number of random numbers
    int initial_seed = 42; // Initial seed for the LCG
    int min_val = -10;     // Minimum value of random numbers
    int max_val = 10;      // Maximum value of random numbers

    // Timing the function
    auto start_time = std::chrono::high_resolution_clock::now();
    int result = total_max_subarray_sum(n, initial_seed, min_val, max_val);
    auto end_time = std::chrono::high_resolution_clock::now();

    std::cout << "Total Maximum Subarray Sum (20 runs): " << result << std::endl;
    std::cout << "Execution Time: " << std::setprecision(6) << std::fixed << std::chrono::duration<double>(end_time - start_time).count() << " seconds" << std::endl;

    return 0;
}