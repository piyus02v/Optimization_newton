#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <utility>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Generate a random number between a and b
double random_double(double a, double b)
{
    random_device rd;
    mt19937 rng(rd());
    uniform_real_distribution<double> dist(a, b);
    return dist(rng);
}

// Function to minimize (Sum of Squares)
double func(const VectorXd &x)
{
    double result = 0.0;
    for (int i = 0; i < x.size(); ++i)
    {
        result += (i + 1) * x[i] * x[i];
    }
    return result;
}

// Gradient of the Sum of Squares function
VectorXd gradient(const VectorXd &x)
{
    VectorXd grad(x.size());
    for (int i = 0; i < x.size(); ++i)
    {
        grad[i] = 2 * (i + 1) * x[i];
    }
    return grad;
}

// Hessian of the Sum of Squares function
MatrixXd hessian(const VectorXd &x)
{
    MatrixXd hess = MatrixXd::Zero(x.size(), x.size());
    for (int i = 0; i < x.size(); ++i)
    {
        hess(i, i) = 2 * (i + 1);
    }
    return hess;
}

// Bounding Phase Method for unidirectional search
pair<double, double> bounding_phase(double lb, double ub, double delta, const VectorXd &x, const VectorXd &p)
{
    double alpha = random_double(lb, ub); // Initial random point
    cout << "Starting value: " << alpha << endl;

    double dx = delta;
    double alpha_lb = max(alpha - dx, lb);
    double alpha_ub = min(alpha + dx, ub);
    double f_alpha = func(x + alpha * p), f_alpha_lb = func(x + alpha_lb * p), f_alpha_ub = func(x + alpha_ub * p);
    int k = 0;

    while (alpha_ub <= ub && alpha_lb >= lb)
    {
        dx = (f_alpha <= f_alpha_lb && f_alpha >= f_alpha_ub) ? abs(dx) : -abs(dx);

        k++;
        double alpha_new = alpha + pow(2, k) * dx;
        if (alpha_new > ub)
            alpha_new = ub;
        if (alpha_new < lb)
            alpha_new = lb;

        if (func(x + alpha_new * p) > f_alpha)
        {
            return {alpha, alpha_new};
        }

        alpha = alpha_new;
        alpha_ub = min(alpha + abs(dx), ub);
        alpha_lb = max(alpha - abs(dx), lb);
        f_alpha = func(x + alpha * p);
        f_alpha_lb = func(x + alpha_lb * p);
        f_alpha_ub = func(x + alpha_ub * p);
    }

    return {alpha, alpha};
}

// Secant Method for finding the optimal step size
double secant_method(double alpha1, double alpha2, double epsilon, const VectorXd &x, const VectorXd &p)
{
    double z;
    int max_iter = 100; // Limit the number of iterations
    int iter = 0;

    while (iter < max_iter)
    {
        // Compute directional derivatives at alpha1 and alpha2
        double grad_alpha1 = gradient(x + alpha1 * p).dot(p); // directional derivative at alpha1
        double grad_alpha2 = gradient(x + alpha2 * p).dot(p); // directional derivative at alpha2

        // Prevent division by zero
        if (abs(grad_alpha2 - grad_alpha1) < 1e-9)
        {
            cerr << "Warning: Small denominator in Secant Method, stopping early." << endl;
            break;
        }

        // Secant update step for alpha
        z = alpha2 - grad_alpha2 * ((alpha2 - alpha1) / (grad_alpha2 - grad_alpha1));

        // Check for convergence (small gradient)
        if (abs(gradient(x + z * p).dot(p)) <= epsilon)
            break;

        // Update alpha values based on the result
        if (gradient(x + z * p).dot(p) < grad_alpha2)
        {
            alpha1 = alpha2;
            alpha2 = z;
        }
        else
        {
            alpha2 = z;
        }

        iter++;
    }

    return z;
}

VectorXd newtons_method(VectorXd x, double epsilon, double delta, const vector<pair<double, double>> &bounds)
{
    while (true)
    {
        VectorXd grad = gradient(x);
        MatrixXd hess = hessian(x);

        // Check for convergence
        if (grad.norm() < epsilon)
            break;

        // Compute the Newton direction: p = -H^-1 * grad
        VectorXd p = -hess.inverse() * grad;

        // Debugging output for current state
        cout << "Current x: " << x.transpose() << endl;
        cout << "Current gradient: " << grad.transpose() << endl;
        cout << "Current direction (p): " << p.transpose() << endl;

        // Determine valid alpha range based on bounds of x
        double alpha_lb = -std::numeric_limits<double>::infinity(); // Start from -infinity for lower bound
        double alpha_ub = std::numeric_limits<double>::infinity();  // Start from +infinity for upper bound

        for (int i = 0; i < x.size(); ++i)
        {
            if (p[i] > 0)
            {
                alpha_ub = std::min(alpha_ub, (bounds[i].second - x[i]) / p[i]);
                alpha_lb = std::max(alpha_lb, (bounds[i].first - x[i]) / p[i]);
            }
            else if (p[i] < 0)
            {
                alpha_lb = std::max(alpha_lb, (bounds[i].second - x[i]) / p[i]);
                alpha_ub = std::min(alpha_ub, (bounds[i].first - x[i]) / p[i]);
            }
        }

        // Debugging output for alpha bounds
        cout << "Calculated alpha bounds: [" << alpha_lb << ", " << alpha_ub << "]" << endl;

        // Ensure bounds are valid
        if (alpha_lb > alpha_ub)
        {
            cout << "Invalid alpha bounds: [" << alpha_lb << ", " << alpha_ub << "]. Adjusting bounds." << endl;
            double t = alpha_lb;
            alpha_lb = alpha_ub; // Reset to a valid value if invalid
            alpha_ub = t;
            cout << "Re-Calculated alpha bounds: [" << alpha_lb << ", " << alpha_ub << "]" << endl;
        }

        // Perform unidirectional search to find alpha
        auto alpha_bounds = bounding_phase(alpha_lb, alpha_ub, delta, x, p);
        cout << "Bounding phase result: [" << alpha_bounds.first << ", " << alpha_bounds.second << "]" << endl;

        // Refine alpha using Secant Method
        double alpha = secant_method(alpha_bounds.first, alpha_bounds.second, epsilon, x, p);
        cout << "Secant method result (alpha): " << alpha << endl;

        // Update x: x = x + alpha * p
        x += alpha * p;
    }
    return x;
}

int main()
{
    int n = 5;                              // Number of variables (dimension of the problem)
    vector<pair<double, double>> bounds(n); // Lower and upper bounds for each variable
    VectorXd x(n);                          // Initial random point
    double epsilon, delta;

    // Input bounds for each variable (hypercube range)
    cout << "Enter the lower and upper bounds for each variable:" << endl;
    for (int i = 0; i < n; ++i)
    {
        cout << "Variable " << i + 1 << " lower bound: ";
        cin >> bounds[i].first;
        cout << "Variable " << i + 1 << " upper bound: ";
        cin >> bounds[i].second;
    }

    // Generate a random initial point within the bounds
    cout << "Initial random point: ";
    for (int i = 0; i < n; ++i)
    {
        x[i] = random_double(bounds[i].first, bounds[i].second);
        cout << x[i] << " ";
    }
    cout << endl;

    // Input parameters epsilon and delta
    cout << "Enter epsilon (convergence tolerance) and delta (step size for bounding phase): ";
    cin >> epsilon >> delta;

    // Perform Newton's method
    VectorXd result = newtons_method(x, epsilon, delta, bounds);

    // Output final result
    cout << "Final result: ";
    for (int i = 0; i < result.size(); ++i)
    {
        cout << result[i] << " ";
    }
    cout << endl;

    return 0;
}
