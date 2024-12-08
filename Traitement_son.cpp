#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <complex>
#include <fftw3.h>


#define M_PI 3.14159265358979323846


// Définition de la fonction du filtre Butterworth
int filterButterworthOrder3(double *x, double *y, int N, int fc, int fe) {
    double alpha = M_PI * (double)fc / (double)fe;  // Calcul de la fréquence normalisée
    double A = +1.0 + 2.0 * alpha + 2.0 * alpha * alpha + 1.0 * alpha * alpha * alpha;
    double B = -3.0 - 2.0 * alpha + 2.0 * alpha * alpha + 3.0 * alpha * alpha * alpha;
    double C = +3.0 - 2.0 * alpha - 2.0 * alpha * alpha + 3.0 * alpha * alpha * alpha;
    double D = -1.0 + 2.0 * alpha - 2.0 * alpha * alpha + 1.0 * alpha * alpha * alpha;

    double a0 = 1.0 * alpha * alpha * alpha / A;
    double a1 = 3.0 * alpha * alpha * alpha / A;
    double a2 = 3.0 * alpha * alpha * alpha / A;
    double a3 = 1.0 * alpha * alpha * alpha / A;
    double b1 = -B / A;
    double b2 = -C / A;
    double b3 = -D / A;

    // Application du filtre
    for (int i = 0; i < N; i++) {
        if (i == 0) {
            y[i] = a0 * x[i];
        } else if (i == 1) {
            y[i] = a1 * x[i - 1] + a0 * x[i];
        } else if (i == 2) {
            y[i] = a2 * x[i - 2] + a1 * x[i - 1] + a0 * x[i];
        } else {
            y[i] = a3 * x[i - 3] + a2 * x[i - 2] + a1 * x[i - 1] + a0 * x[i];
        }
    }
    return 0;
}

// Lecture des données audio depuis un fichier texte
std::vector<double> readAudioData(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<double> data;
    double value;

    if (file.is_open()) {
        while (file >> value) {
            data.push_back(value);
        }
    } else {
        std::cerr << "Impossible d'ouvrir le fichier " << filename << std::endl;
    }

    return data;
}

// Sauvegarde des données traitées dans un fichier texte
void saveDataToFile(const std::string& filename, const std::vector<double>& data) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (const double& value : data) {
            file << value << std::endl;
        }
    } else {
        std::cerr << "Impossible de sauvegarder le fichier " << filename << std::endl;
    }
}

// Transformée de Fourier rapide (FFT), conservation uniquement de la partie des fréquences positives
std::vector<std::complex<double>> performFFT(const std::vector<double>& data) {
    int N = data.size();
    std::vector<std::complex<double>> fft_result(N);

    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    // Remplissage des données d'entrée
    for (int i = 0; i < N; ++i) {
        in[i][0] = data[i];
        in[i][1] = 0.0;
    }

    fftw_execute(plan);  // Exécution de la FFT

    // Conservation uniquement de la partie des fréquences positives
    int half_N = N / 2;
    for (int i = 0; i < half_N; ++i) {
        fft_result[i] = std::complex<double>(out[i][0], out[i][1]);
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return fft_result;
}

// Transformée de Fourier inverse (Inverse FFT)
std::vector<double> performInverseFFT(const std::vector<std::complex<double>>& fft_data) {
    int N = fft_data.size();
    std::vector<double> result(N);

    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Remplissage des données d'entrée
    for (int i = 0; i < N; ++i) {
        in[i][0] = fft_data[i].real();
        in[i][1] = fft_data[i].imag();
    }

    fftw_execute(plan);  // Exécution de l'inverse FFT

    // Extraction des résultats
    for (int i = 0; i < N; ++i) {
        result[i] = out[i][0] / N;  // Normalisation
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return result;
}

// Application de la fenêtre de Hamming
void applyHammingWindow(std::vector<double>& data) {
    int N = data.size();
    for (int i = 0; i < N; ++i) {
        data[i] *= 0.54 - 0.46 * cos(2 * M_PI * i / (N - 1));
    }
}

int main() {
    // Lecture des données audio
    std::vector<double> audio_data = readAudioData("selected_audio_info.txt");
    int N = audio_data.size();

    // Lecture de la fréquence d'échantillonnage
    int fe = 44100;  // Valeur par défaut
    std::ifstream fe_file("sample_rate.txt");
    if (fe_file.is_open()) {
        fe_file >> fe;  // Lecture de la fréquence d'échantillonnage depuis le fichier
        fe_file.close();
    } else {
        std::cerr << "Impossible de lire le fichier de la fréquence d'échantillonnage, utilisation de la valeur par défaut 44100 Hz" << std::endl;
    }

    // Lecture de la fréquence de coupure
    int fc = 300;  // Valeur par défaut
    std::ifstream fc_file("cutoff_frequency.txt");
    if (fc_file.is_open()) {
        fc_file >> fc;  // Lecture de la fréquence de coupure depuis le fichier
        fc_file.close();
    } else {
        std::cerr << "Impossible de lire le fichier de la fréquence de coupure, utilisation de la valeur par défaut 300 Hz" << std::endl;
    }

    // Création du filtre et application
    std::vector<double> filtered_data(N);
    filterButterworthOrder3(audio_data.data(), filtered_data.data(), N, fc, fe);

    // Application de la fenêtre de Hamming
    applyHammingWindow(filtered_data);

    // Sauvegarde des données traitées (filtrées et avec fenêtre de Hamming)
    saveDataToFile("filtered_audio_info.txt", filtered_data);

    // Exécution de la FFT, conservation uniquement de la partie des fréquences positives
    auto fft_result = performFFT(filtered_data);
    
    // Sauvegarde des résultats FFT (partie des fréquences positives)
    std::ofstream fft_file("fft_result.txt");
    for (const auto& complex_num : fft_result) {
        fft_file << complex_num.real() << " " << complex_num.imag() << std::endl;
    }

    // Exécution de l'inverse FFT
    auto inverse_fft_result = performInverseFFT(fft_result);

    // Sauvegarde des résultats de l'inverse FFT
    saveDataToFile("inverse_fft_result.txt", inverse_fft_result);

    std::cout << "Traitement terminé !" << std::endl;

    return 0;
}
