#include "Processing.h"

#include "kiss_fft.h"

template<typename T>
T min(const T& a, const T& b) { return a < b ? a : b; }
template<typename T>
T max(const T& a, const T& b) { return a < b ? b : a; }

// Hann window : equals 1 in 0, and 0 in 1
double window(double x) { return 0.5*(1 - cos(2*M_PI*x)); }

Audio vocoder(const Audio& modulator, const Audio& frequencies, int bands) {

	//assert(modulator.sampleRate == frequencies.sampleRate);

	// Output signal
	Audio dst;
	dst.sampleRate = frequencies.sampleRate;
	int n =  (modulator.samples.size(), frequencies.samples.size());
	dst.samples = std::vector<double>(n);

	// coeffs added to the output
	std::vector<double> coeffs(n, 0);

	// FFT parameters
	int fftSize = 512;
	auto cfg = kiss_fft_alloc(fftSize, false, 0, 0);
	auto cfgI = kiss_fft_alloc(fftSize, true, 0, 0);

	// Overlaping windows
	for (int i = 0; i < n - fftSize / 2; i+= fftSize/2) {

		std::vector<kiss_fft_cpx> input(fftSize), output(fftSize);
		for (int j = 0; j < fftSize; j++) {

			// weighting the samples
			double coeff = window(double(j) / fftSize);
			input[j] = { float(coeff * modulator.samples[i + j]), 0 };
		}

		// forward FFT
		kiss_fft(cfg, input.data(), output.data());

		// filtering
		for (int i = 0; i < 0.1*fftSize / 2; i++) {
			output[i] = { 0, 0 };
			output[fftSize - 1 - i] = { 0, 0 };
		}

		// backward FFT
		kiss_fft(cfgI, output.data(), input.data());

		for (int j = 0; j < min(fftSize,n-i); j++) {

			double coeff = window(double(j) / fftSize);
			dst.samples[i + j] += coeff * input[j].r / fftSize;
			coeffs[i + j] += coeff*coeff;
		}
	}

	// Normalising samples by the added coefficients
	for (int i = 0; i < n; i++) {
		double coeff = max(coeffs[i],1E-5);
		dst.samples[i] /= coeff;
	}

	return dst;
}