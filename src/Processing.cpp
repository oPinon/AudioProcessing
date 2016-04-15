#include "Processing.h"


#include "kiss_fft.h"

#include <opencv2\opencv.hpp>

template<typename T>
T min(const T& a, const T& b) { return a < b ? a : b; }
template<typename T>
T max(const T& a, const T& b) { return a < b ? b : a; }

// Hann window : equals 1 in 0, and 0 in 1
double window(double x) { return 0.5*(1 - cos(2 * M_PI*x)); }

Audio vocoder(const Audio& modulator, const Audio& frequencies, int nbBands) {

	//assert(modulator.sampleRate == frequencies.sampleRate);

	// Output signal
	Audio dst;
	dst.sampleRate = frequencies.sampleRate;
	int n = min(modulator.samples.size(), frequencies.samples.size());
	dst.samples = std::vector<double>(n);

	// coeffs added to the output
	std::vector<double> coeffs(n, 0);

	// FFT parameters
	int fftSize = 256;
	auto cfg = kiss_fft_alloc(fftSize, false, 0, 0);
	auto cfgI = kiss_fft_alloc(fftSize, true, 0, 0);

	// Overlaping windows
	for (int i = 0; i < n - fftSize / 2; i += fftSize / 2) {

		std::vector<kiss_fft_cpx> input(fftSize), output(fftSize),
			input2(fftSize), output2(fftSize);
		for (int j = 0; j < fftSize; j++) {

			// weighting the samples
			double coeff = window(double(j) / fftSize);
			input[j] = { float(coeff * frequencies.samples[i + j]), 0 };
			input2[j] = { float(coeff * modulator.samples[i + j]), 0 };
		}

		// forward FFT
		kiss_fft(cfg, input.data(), output.data());
		kiss_fft(cfg, input2.data(), output2.data());

		// filtering
		{
			// extracting the band amplitudes
			std::vector<double> bands(nbBands);
			int bandSize = fftSize / 2 / nbBands;
			for (int b = 0; b < nbBands; b++) {
				for (int f = 0; f < bandSize; f++) {
					{ // cos component
						double r = output2[b*bandSize + f].r;
						double i = output2[b*bandSize + f].i;
						bands[b] += sqrt(r*r + i*i);
					}
					{ // sin component
						double r = output2[fftSize - 1 - (b*bandSize + f)].r;
						double i = output2[fftSize - 1 - (b*bandSize + f)].i;
						bands[b] += sqrt(r*r + i*i);
					}
				}
				bands[b] /= (2 * bandSize);
			}

			// normalizing the bands TODO ?
			double maxB = 1E-5;
			for (double b : bands) { if (b>maxB) { maxB = b; } }
			for (double& b : bands) { b /= sqrt(fftSize); }

			// displaying the bands
#if 0
			if (i % 1024 == 0) {
				int h = 256;
				cv::Mat histogram(cv::Size(bands.size(), h), CV_8UC1);
				for (int i = 0; i < bands.size(); i++) {
					for (int j = 0; j < h; j++) {
						int v = 0;
						if (bands[i] * h > h - j) { v = 255; }
						histogram.data[j*bands.size() + i] = v;
					}
				}
				cv::resize(histogram, histogram, cv::Size(512, h), 0, 0, cv::INTER_NEAREST);
				cv::imshow("sound", histogram); cv::waitKey(16);
			}
#endif

			// weighting the frequencies vector
			for (int i = 0; i < fftSize / 2; i++) {
				int band = i / bandSize;
				double weight = bands[band];

				output[i].r *= weight;
				output[i].i *= weight;
				output[fftSize - 1 - i].r *= weight;
				output[fftSize - 1 - i].i *= weight;
			}
		}

		// backward FFT
		kiss_fft(cfgI, output.data(), input.data());
		kiss_fft(cfgI, output2.data(), input2.data());

		for (int j = 0; j < min(fftSize, n - i); j++) {

			double coeff = window(double(j) / fftSize);
			dst.samples[i + j] += coeff * input[j].r / fftSize;
			coeffs[i + j] += coeff*coeff;
		}
	}

	// Normalising samples by the added coefficients
	for (int i = 0; i < n; i++) {
		double coeff = max(coeffs[i], 1E-5);
		dst.samples[i] /= coeff;
	}

	return dst;
}

void Audio::spectrogram(char* fileName, int fftSize) {

	// FFT parameters
	auto cfg = kiss_fft_alloc(fftSize, false, 0, 0);

	// Destination image
	int nbFFT = this->samples.size() / fftSize * 2 - 1;
	cv::Mat dst(cv::Size(nbFFT, fftSize / 2), CV_64FC1);
	double* dstP = (double*)dst.data;

	for (int i = 0; i < this->samples.size() - fftSize; i += fftSize / 2) {

		// temporal input (weighted)
		std::vector<kiss_fft_cpx> input(fftSize), output(fftSize);
		for (int j = 0; j < fftSize; j++) {
			double coeff = window(double(j) / fftSize);
			input[j] = { kiss_fft_scalar(coeff * samples[i + j]), 0 };
		}

		// forward FFT
		kiss_fft(cfg, input.data(), output.data());

		// addind to spectrogram
		for (int j = 0; j < fftSize / 2; j++) {
			double value = 0;
			{ // cos component
				double r = output[j].r;
				double i = output[j].i;
				value += r*r + i*i;
			}
			{ // sin component
				double r = output[fftSize - 1 - j].r;
				double i = output[fftSize - 1 - j].i;
				value += r*r + i*i;
			}
			value = sqrt(value / (2 * fftSize));

			// coloring the pixel
			dstP[nbFFT*(fftSize / 2 - 1 - j) + i / (fftSize / 2)] = value;
		}
	}

	// Tuning and colormaping the image for display
	cv::pow(dst, 0.3, dst);
	//cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
	dstP = (double*)dst.data;
	cv::Mat dstCol(dst.size(), CV_8UC3);
	for (int i = 0; i < dstCol.size().width*dstCol.size().height; i++) {

		double pos = min<double>(1, max<double>(0, dstP[i]));
		const static float blueHue = 360, yellowHue = 180;

		float saturation = 1 - pow(pos, 15);
		float value = pos;
		float hue = blueHue * (1 - pos) + yellowHue * pos;

		float C = value * saturation;
		float X = C * (1 - abs(fmod((hue / 60), 2) - 1));
		float m = value - C;
		float r, g, b;
		if (0 <= hue && hue <= 60) { r = C; g = X; b = 0; }
		else if (60 <= hue && hue <= 120) { r = X; g = C; b = 0; }
		else if (120 <= hue && hue <= 180) { r = 0; g = C; b = X; }
		else if (180 <= hue && hue <= 240) { r = 0; g = X; b = C; }
		else if (240 <= hue && hue <= 300) { r = X; g = 0; b = C; }
		else if (300 <= hue && hue <= 360) { r = C; g = 0; b = X; }

		dstCol.data[3 * i + 2] = 255 * (b + m);
		dstCol.data[3 * i + 1] = 255 * (g + m);
		dstCol.data[3 * i + 0] = 255 * (r + m);
	}

	// writing to file
	cv::imwrite(fileName, dstCol);
}