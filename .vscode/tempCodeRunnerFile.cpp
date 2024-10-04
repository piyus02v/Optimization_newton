#include <iostream>
#include <vector>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main() {
    // x and y values
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {1, 4, 9, 16, 25};

    // Plot x and y with red dots
    plt::scatter(x, y, 10.0, {{"color", "red"}});

    // Display the plot
    plt::show();

    return 0;
}