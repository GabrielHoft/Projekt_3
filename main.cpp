#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <iomanip>
#include <matplot/matplot.h>

namespace pyt = pybind11;
using namespace std;
using namespace matplot;

#ifndef pi
#define pi 3.14159265358979323846
#endif

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

void wyswietlanie_sygnalu(const vector<double>& czas, const vector<double>& sygnal, const string& nazwa = "") {
    plot(czas, sygnal);
    if (!nazwa.empty()) {
        title(nazwa);
    }
    xlabel("czas [s]");
    show();
}

pair<vector<double>, vector<double>> generowanie_sin(double czestotliwosc, double amplituda, double okres, double samplowanie) {
    int nr_sampla = okres * samplowanie;
    vector<double> sygnal(nr_sampla);
    vector<double> czas(nr_sampla);
    double dt = 1.0 / samplowanie;
    for (int i = 0; i < nr_sampla; ++i) {
        czas[i] = i * dt;
        sygnal[i] = amplituda * sin(2 * pi * czestotliwosc * czas[i]);
    }
    return { czas, sygnal };
}

pair<vector<double>, vector<double>> generowanie_cos(double czestotliwosc, double amplituda, double okres, double samplowanie) {
    int nr_sampla = okres * samplowanie;
    vector<double> sygnal(nr_sampla);
    vector<double> czas(nr_sampla);
    double dt = 1.0 / samplowanie;
    for (int i = 0; i < nr_sampla; ++i) {
        czas[i] = i * dt;
        sygnal[i] = amplituda * cos(2 * pi * czestotliwosc * czas[i]);
    }
    return { czas, sygnal };
}

pair<vector<double>, vector<double>> generowanie_pros(double czestotliwosc, double amplituda, double okres, double samplowanie) {
    int nr_sampla = okres * samplowanie;
    vector<double> sygnal(nr_sampla);
    vector<double> czas(nr_sampla);
    double dt = 1.0 / samplowanie;
    for (int i = 0; i < nr_sampla; ++i) {
        czas[i] = i * dt;
        sygnal[i] = amplituda * copysign(1.0, sin(2 * pi * czestotliwosc * czas[i]));
    }
    return { czas, sygnal };
}

pair<vector<double>, vector<double>> generowanie_pilo(double czestotliwosc, double amplituda, double okres, double samplowanie) {
    int nr_sampla = okres * samplowanie;
    vector<double> sygnal(nr_sampla);
    vector<double> czas(nr_sampla);
    double dt = 1.0 / samplowanie;
    for (int i = 0; i < nr_sampla; ++i) {
        czas[i] = i * dt;
        sygnal[i] = (2.0 * amplituda / pi) * atan(1.0 / tan(pi * czestotliwosc * czas[i]));
    }
    return { czas, sygnal };
}

vector<float> generowanie_kernela_gaussa_1D(float sigma) {
    int rozmiar_kernela = 2 * ceil(3 * sigma) + 1;
    vector<float> kernel(rozmiar_kernela);
    float suma = 0.0;
    int promien_kernela = rozmiar_kernela / 2;
    for (int i = -promien_kernela; i <= promien_kernela; ++i) {
        kernel[i + promien_kernela] = exp(-(i * i) / (2 * sigma * sigma)) / (sqrt(2 * pi) * sigma);
        suma += kernel[i + promien_kernela];
    }
    for (int i = 0; i < rozmiar_kernela; ++i) {
        kernel[i] /= suma;
    }
    return kernel;
}

vector<vector<double>> generowanie_kernela_gaussa_2D(double sigma) {
    int promien_kernela = ceil(3 * sigma);
    int rozmiar_kernela = 2 * promien_kernela + 1;
    vector<vector<double>> kernel(rozmiar_kernela, vector<double>(rozmiar_kernela));
    double suma = 0.0;
    double s = 2.0 * sigma * sigma;
    for (int x = -promien_kernela; x <= promien_kernela; ++x) {
        for (int y = -promien_kernela; y <= promien_kernela; ++y) {
            double r = sqrt(x * x + y * y);
            kernel[x + promien_kernela][y + promien_kernela] = (exp(-(r * r) / s)) / (pi * s);
            suma += kernel[x + promien_kernela][y + promien_kernela];
        }
    }
    for (int i = 0; i < rozmiar_kernela; ++i) {
        for (int j = 0; j < rozmiar_kernela; ++j) {
            kernel[i][j] /= suma;
        }
    }
    return kernel;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "przetwarzanie sygnalow";
    m.def("wyswietlanie_sygnalu", &wyswietlanie_sygnalu, pyt::arg("time"), pyt::arg("signal"), pyt::arg("title") = "");
    m.def("generowanie_sin", &generowanie_sin);
    m.def("generowanie_cos", &generowanie_cos);
    m.def("generowanie_pros", &generowanie_pros);
    m.def("generowanie_pilo", &generowanie_pilo);
    m.def("generowanie_kernela_gaussa_1D", &generowanie_kernela_gaussa_1D);
    m.def("generowanie_kernela_gaussa_2D", &generowanie_kernela_gaussa_2D);
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}