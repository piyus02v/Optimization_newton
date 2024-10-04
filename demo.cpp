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
double func(const VectorXd &x,int qno)
{
    double result=0.0;
    switch(qno)
    {
        case 1:{
            
            for (int i = 0; i < x.size(); ++i)
                {
                    result += (i + 1) * x[i] * x[i];
                }
            }break;
        case 2:{
            for (int i = 0; i < x.size()-1; ++i)
                {
                    result += 100*pow((x[i+1]-x[i]*x[i]),2)+pow(x[i]-1,2);
                }
            }break;
        case 3:{
            result=(x[0]-1)*(x[0]-1);
            for (int i = 1; i < x.size(); ++i)
                {
                    result += (i+1)*pow((2*x[i]*x[i]-x[i-1]),2);
                }
        }break;
        case 4:{
            double sum1=0.0,sum2=0.0;
            for (int i = 0; i < x.size(); ++i)
                {
                    sum1 += (x[i]-1)*(x[i]-1);
                }
            for (int i = 1; i < x.size(); ++i)
                {
                    sum2 += x[i]*x[i-1];
                }
                result=sum1-sum2;
        }break;
        case 5:{
            double sum1=0.0,sum2=0.0;
            for (int i = 0; i < x.size(); ++i)
                {
                    sum1 += x[i]*x[i];
                }
            for (int i = 0; i < x.size(); ++i)
                {
                    sum2 += 0.5*(i+1)*x[i];
                }
            
            result=sum1+pow(sum2,2)+pow(sum2,4);
            
        }break;
        default:{
            cout<<"Wrong input"<<endl;

        }break;
    }
    
    return result;
}

// Gradient of the Sum of Squares function
VectorXd gradient(const VectorXd& x,int qno) {
    size_t n = x.size(); // Dimension of input vector
    VectorXd gradient(n);
    double delta=1e-5; // To store the gradient

    // Loop over each dimension
    for (size_t i = 0; i < n; ++i) {
        VectorXd x_plus = x;
        VectorXd x_minus = x;
        
        // Increment and decrement the i-th component by delta
        x_plus[i] += delta;
        x_minus[i] -= delta;
        
        // Compute the function value at x_plus and x_minus
        double f_plus = func(x_plus,qno);
        double f_minus = func(x_minus,qno);
        
        // Approximate the gradient for the i-th component
        gradient[i] = (f_plus - f_minus) / (2 * delta);
    }
    
    return gradient;
}
// Hessian of the Sum of Squares function
MatrixXd hessian(const VectorXd& x,int qno) {
    size_t n = x.size(); // Dimension of input vector
    MatrixXd hessian(n, n); // To store the Hessian matrix
    hessian.setZero();
    double delta=1e-5; // Initialize to zero

    // Loop over each pair of dimensions (i, j)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) { // Hessian is symmetric, so only compute for j <= i
            VectorXd x_plusplus = x;
            VectorXd x_plusminus = x;
            VectorXd x_minusplus = x;
            VectorXd x_minusminus = x;

            // Increment and decrement the i-th and j-th components
            x_plusplus[i] += delta; x_plusplus[j] += delta;
            x_plusminus[i] += delta; x_plusminus[j] -= delta;
            x_minusplus[i] -= delta; x_minusplus[j] += delta;
            x_minusminus[i] -= delta; x_minusminus[j] -= delta;

            // Compute the function values for the four modified vectors
            double f_plusplus = func(x_plusplus,qno);
            double f_plusminus = func(x_plusminus,qno);
            double f_minusplus = func(x_minusplus,qno);
            double f_minusminus = func(x_minusminus,qno);

            // Approximate the second derivative for (i, j) using central difference
            hessian(i, j) = hessian(j, i) = (f_plusplus - f_plusminus - f_minusplus + f_minusminus) / (4 * delta * delta);
        }
    }

    return hessian;
}
//MatrixXd
// Bounding Phase Method for unidirectional search
pair<double, double> bounding_phase(double lb, double ub, double delta, const VectorXd &x, const VectorXd &p,int qno)
{
    double alpha = random_double(lb, ub); // Initial random point
    cout << "Starting value: " << alpha << endl;

    double dx = delta;
    double alpha_lb = max(alpha - dx, lb);
    double alpha_ub = min(alpha + dx, ub);
    double f_alpha = func(x + alpha * p,qno), f_alpha_lb = func(x + alpha_lb * p,qno), f_alpha_ub = func(x + alpha_ub * p,qno);
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

        if (func(x + alpha_new * p,qno) > f_alpha)
        {
            return {alpha, alpha_new};
        }

         alpha = alpha_new;
        alpha_ub = min(alpha + abs(dx), ub);
        alpha_lb = max(alpha - abs(dx), lb);
        f_alpha = func(x + alpha * p,qno);
        f_alpha_lb = func(x + alpha_lb * p,qno);
        f_alpha_ub = func(x + alpha_ub * p,qno);
    }

    return {alpha, alpha};
}

// Secant Method for finding the optimal step size
double secant_method(double alpha1, double alpha2, double epsilon, const VectorXd &x, const VectorXd &p,int qno)
{
    double z;
    int max_iter = 100; // Limit the number of iterations
    int iter = 0;

    while (iter < max_iter)
    {
        // Compute directional derivatives at alpha1 and alpha2
        double grad_alpha1 = gradient(x + alpha1 * p,qno).dot(p); // directional derivative at alpha1
        double grad_alpha2 = gradient(x + alpha2 * p,qno).dot(p); // directional derivative at alpha2

        // Prevent division by zero
        if (abs(grad_alpha2 - grad_alpha1) < 1e-9)
        {
            cerr << "Warning: Small denominator in Secant Method, stopping early." << endl;
            break;
        }

        // Secant update step for alpha
        z = alpha2 - grad_alpha2 * ((alpha2 - alpha1) / (grad_alpha2 - grad_alpha1));

        // Check for convergence (small gradient)
        if (abs(gradient(x + z * p,qno).dot(p)) <= epsilon)
            break;

        // Update alpha values based on the result
        if (gradient(x + z * p,qno).dot(p) < grad_alpha2)
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

VectorXd newtons_method(VectorXd x, double epsilon, double delta, const vector<pair<double, double>> &bounds,int qno)
{
    while (true)
    {
        VectorXd grad = gradient(x,qno);
        MatrixXd hess = hessian(x,qno);

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
        auto alpha_bounds = bounding_phase(alpha_lb, alpha_ub, delta, x, p,qno);
        cout << "Bounding phase result: [" << alpha_bounds.first << ", " << alpha_bounds.second << "]" << endl;

        // Refine alpha using Secant Method
        double alpha = secant_method(alpha_bounds.first, alpha_bounds.second, epsilon, x, p,qno);
        cout << "Secant method result (alpha): " << alpha << endl;

        // Update x: x = x + alpha * p
        x += alpha * p;
    }
    return x;
}

int main()
{
                                 
     
                              
    double epsilon, delta;
    int qno;
    cout<<"Enter question number:"<<endl;
    cin>>qno;
    int n;// Number of variables (dimension of the problem)
    cout<<"Enter dimension d:"<<endl;
    cin>>n;
    vector<pair<double, double>> bounds(n);
    VectorXd x(n);
    // Input bounds for each variable (hypercube range)
    cout << "Enter the lower and upper bounds for each variable:" << endl;// Lower and upper bounds for each variable
    double lb,ub;
    cin>>lb>>ub;
    for (int i = 0; i < n; ++i)
    {
        bounds[i].first=lb;
        bounds[i].second=ub;
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
    VectorXd result = newtons_method(x, epsilon, delta, bounds,qno);

    // Output final result
    cout << "Final result: ";
    for (int i = 0; i < result.size(); ++i)
    {
        cout << result[i] << " ";
    }
    cout << endl;

    return 0;
}
    

     
     

      

      
      






      

 
	