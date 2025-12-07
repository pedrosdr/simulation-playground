# include <iostream>
# include <math.h>
# include <vector>

std::vector<float> flatten(const std::vector<std::vector<float>> &x) {
    size_t N1 = x.size();
    size_t N2 = x[0].size();

    std::vector<float> y(0.0f, N1*N2);

    size_t ii = 0;
    for(int i = 0; i < N1; i++) {
        for(int j = 0; j < N2; j++) {
            y[ii] = x[i][j];
            ii++;
        }
    }
    return y;
}

std::vector<float> linspace(float from, float to, size_t size) {
    std::vector<float> x(size, 0.0f);
    x[0] = from;
    float dx = (to-from)/((float)size-1);
    for(int i = 1; i < size; i++) {
        x[i] = x[i-1]+dx;
    }
    return x;
}

float mean(const std::vector<float> &x) {
    size_t n = x.size();
    float reduced = 0.0;
    for(float f : x) {
        reduced += f;
    }
    return (1.0f/(float)n)*reduced;
}

float trapz(const std::vector<float> &y, const std::vector<float> &x) {
    float N = (float)x.size();
    float reduced = 0.0f;
    for (int i = 1; i < N; i++) {
        float dx = x[i] - x[i-1];
        float h = (y[i]+y[i-1])/2.0f;
        reduced += dx*h;
    }
    return reduced;
}

std::vector<float> norm(const std::vector<float> &x, float loc, float scale) {
    size_t N = x.size();
    float pi = 3.14159265358979323846f;
    float A = 1.0f/(scale*sqrt(2.0f*pi));
    std::vector<float> y(N, 0);
    for(int i = 0; i < N; i++) {
        float arg = -0.5*pow((x[i]-loc)/scale, 2.0f);
        float B = exp(arg);
        y[i] = A*B;
    }
    return y;
}

std::vector<float> norm(const std::vector<float> &x) {
    return norm(x, 0.0f, 1.0f);
}

void printv(const std::vector<float> &x) {
    std::cout << "[";
    size_t n = x.size() >= 20? 19 : x.size();
    for(size_t i = 0; i < n; i++) {
        std::printf("%.3f", x[i]);
        if(i != x.size()-1) {
            std::cout << ", ";
        }
    }
    if(n != x.size()) {
        std::cout << "..., " << x[x.size()-1];
    }
    std::cout << "]";
}

void printm(const std::vector<std::vector<float>> &x) {
    std::cout << "[";
    size_t n = x.size() >= 20? 19 : x.size();
    for(size_t i = 0; i < n; i++) {
        if(i > 0) {
            std::cout << " ";
        }
        printv(x[i]);
        if(i != x.size()-1) {
            std::cout << ", \n";
        }
    }
    if(n != x.size()) {
        std::cout << "...,\n ";
        printv(x[x.size()-1]);
    }
    std::cout << "]";
}

int main() {
    std::cout << "Hello world!";
    std::vector<float> x = linspace(-1.96f, 1.96f, 1000);
    std::vector<float> pdf = norm(x, 0, 1);
    std::cout << trapz(pdf, x) << "\n";
    std::vector<std::vector<float>> m = {
        {1.0f, 2.1f, 4.3f},
        {2.2f, 1.3f, 5.0f}
    };
    printm(m);
    printv(flatten(m));
    return 0;
}