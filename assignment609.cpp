#include <bits/stdc++.h>
using namespace std;


 
double random_double(double a, double b) {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(a, b);
    
    return dist(rng);
}
double func(double x)
{
    //return -(pow((2*x-5),4)-pow((x*x-1),3));
    //return -(8+x*x*x-2*x-2*exp(x));
    //return -(4*x*sin(x));
    //return 2*pow(x-3,2)+exp(0.5*x*x);
    //return x*x-10*exp(0.1*x);
    return -(20*sin(x)-15*x*x);
    
}
double fdash(double x)
{
    //return 8*pow((2*x-5),3)-6*x*pow((x*x-1),2);
    //return 3*x*x-2-2*exp(x);
    //return 4*sin(x)+4*x*cos(x);
    //return 4*(x-3)+0.5*x*exp(0.5*x*x);
    //return 2*x-exp(0.1*x);
    return 20*cos(x)-30*x;
    
    
    
}
pair<double, double> bounding_phase(double lb, double ub,double delta) {
    

    // Initial random point
    double x = random_double(lb,ub);
    std::cout << "Starting value: " << x << std::endl;

    double dx = delta;
    double x_lb = x - dx;
    double x_ub = x + dx;
    double f_x = func(x);
    double f_x_lb = func(x_lb);
    double f_x_ub = func(x_ub);
    int k = 0;

    // Ensure that the interval is within bounds
    x_lb = std::max(x_lb, lb);
    x_ub = std::min(x_ub, ub);

    while (x_ub <= ub && x_lb >= lb) {
        if (f_x <= f_x_lb && f_x >= f_x_ub) {
            dx = std::abs(dx);
        } else if (f_x >= f_x_lb && f_x <= f_x_ub) {
            dx = -std::abs(dx);
        }

        k++;
        double x_n = x + std::pow(2, k) * dx;
        double f_x_n = func(x_n);

        if (f_x_n > f_x) {
            if (x_n > ub) x_n = ub;
            if (x_n < lb) x_n = lb;
            return {x, x_n};
        }

        x = x_n;
        x_ub = std::min(x + std::abs(dx), ub);
        x_lb = std::max(x - std::abs(dx), lb);
        f_x = f_x_n;
        f_x_lb = func(x_lb);
        f_x_ub = func(x_ub);
    }

    return {x, x};
}

double secantmethod(double a,double b,double e)
{
    double x1=a;
    double x2=b;
    double z;
    int t=1;
    while(true)
    {
        
        z=x2-fdash(x2)/((fdash(x2)-fdash(x1))/(x2-x1));
        if(abs((fdash(z)))<=e)
        {
            break;
        }
        else if(fdash(z)<0)
        {
            x1=z;
        }
        else
        {
            x2=z;
        }
        

    }
    return z;

}
int main() {
    
    double a,b,e,delta;
    cout<<"Enter upper and lower bounds";
    cin>>a>>b;
    cout<<"enter epsilon and delta";
    cin>>e>>delta;

    
    
    for(int i=1;i<=10;i++)
    {
        cout<<"Run "<<i<<" ->"<<endl;
        cout<<"Limits after bounding phase method->"<<endl;
        pair<double,double> p=bounding_phase(a,b,delta);
        double x=p.first;
        double y=p.second;
        cout<<x<<"  "<<y<<endl;

        cout<<"Applying secant method we get ->";
        cout<<secantmethod(x,y,e)<<endl;
    }


}




