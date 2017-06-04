#include "Processing.h"


#include "kiss_fft.h"
#include <limits>

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

#ifdef AUDIOPROCESSING_USE_OPENCV

#include <opencv2\opencv.hpp>

cv::Mat spect(const std::vector<double>& samples, int fftSize) {

	// FFT parameters
	auto cfg = kiss_fft_alloc(fftSize, false, 0, 0);

	// Destination image
	int nbFFT = samples.size() / fftSize * 2 - 1;
	cv::Mat dst(cv::Size(nbFFT, fftSize / 2), CV_64FC1);
	double* dstP = (double*)dst.data;

	for (int i = 0; i < samples.size() - fftSize; i += fftSize / 2) {

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

	return dst;
}

void Audio::writeImage(const char *fileName) const
{
	size_t fftSize = 1024;

	// FFT parameters
	auto cfg = kiss_fft_alloc(fftSize, false, 0, 0);

	// Destination image
	int nbFFT = samples.size() / fftSize * 2 - 1;
	cv::Mat dst(cv::Size(fftSize, nbFFT), CV_16UC3);
	uint16_t* dstP = (uint16_t*)dst.data;

	for (int i = 0; i < nbFFT; i++)
	{
		// temporal input (weighted)
		std::vector<kiss_fft_cpx> input(fftSize), output(fftSize);
		for (int j = 0; j < fftSize; j++)
		{
			double coeff = window(double(j) / fftSize);
			input[j] = { kiss_fft_scalar(coeff * samples[( i * fftSize ) / 2  + j]), 0 };
		}

		// forward FFT
		kiss_fft(cfg, input.data(), output.data());

		for (int j = 0; j < fftSize; j++)
		{
			size_t y = j;
			size_t x = i;
			const uint16_t maxV = std::numeric_limits<uint16_t>::max();
			dstP[3 * (fftSize * x + y) + 0] = ( maxV * output[j].r / sqrt(fftSize) ) / 2 + maxV / 2;
			dstP[3 * (fftSize * x + y) + 1] = ( maxV * output[j].i / sqrt(fftSize) ) / 2 + maxV / 2;
			dstP[3 * (fftSize * x + y) + 2] = 0;
		}
	}

	cv::imwrite(fileName, dst );
}

void Audio::readImage(const char *fileName)
{
	cv::Mat src = cv::imread(fileName, cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR );
	if (src.type() != CV_16UC3)
	{
		std::cerr << "Audio::readImage : image type isn't 16UC3 (it has "
			<< src.channels() << " channels and a precision of " << src.depth() << ")" << std::endl;
		return;
	}

	size_t fftSize = src.size().width;
	this->samples = std::vector<double>( ( src.size().height + 1 ) * fftSize / 2, 0);
	std::vector<double> coeffs(this->samples.size(), 0);

	// FFT parameters
	auto cfg = kiss_fft_alloc(fftSize, false, 0, 0);

	const uint16_t* srcP = (uint16_t*)src.data;

	for( size_t i = 0; i < src.size().height; i++ )
	{
		std::vector<kiss_fft_cpx> input(fftSize), output(fftSize);
		for( size_t j = 0; j < fftSize; j++ )
		{
			double coeff = window(double(j) / fftSize);
			coeffs[(i * fftSize) / 2 + j] += coeff * coeff;
			input[j].r = srcP[3 * (fftSize * i + j) + 0];
			input[j].i = srcP[3 * (fftSize * i + j) + 1];
		}

		kiss_fft(cfg, input.data(), output.data());

		for (int j = 0; j < fftSize; j++)
		{
			const uint16_t maxV = std::numeric_limits<uint16_t>::max();
			this->samples[(i * fftSize) / 2 + j] += ( output[j].r - maxV / 2 ) / maxV / sqrt(fftSize);
		}
	}

	/*
	for (size_t i = 0; i < this->samples.size(); i++)
		if( coeffs[i] > 1E-3 )
			this->samples[i] /= coeffs[i];
	*/
}

// colors an image in [0;1]
cv::Mat colorMap(const cv::Mat& src) {

	assert(src.depth() == CV_64F);
	double* dstP = (double*)src.data;
	cv::Mat dstCol(src.size(), CV_8UC3);
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
	return dstCol;
}

void Audio::spectrogram(const char* fileName, int fftSize) const {

	cv::Mat dst = spect(this->samples, fftSize);

	// Tuning and colormaping the image for display
	cv::pow(dst, 0.3, dst);
	//cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
	cv::Mat dstCol = colorMap(dst);

	// writing to file
	cv::imwrite(fileName, dstCol);
}

double Audio::period(double minPeriod, double maxPeriod) const {

	bool display = true; // HACK : remove
	std::string dispFolder = "../../data/BPM/";

	// Computing the spectrogram
	int fftSize = 512; // TODO : which size ?
	std::cout << "Computing the spectrogram" << std::endl;
	cv::Mat spectrogram = spect(this->samples, fftSize);
	if (display) {
		cv::Mat dst;
		cv::pow(spectrogram, 0.3, dst);
		cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
		cv::imwrite(dispFolder + "spectrogram.png", colorMap(dst));
	}
	int specH = spectrogram.size().height;
	double timePerFrame = double(fftSize/2) / sampleRate; // time between each FFT

	// Computing the correlation (only for the next frames)
	//double maxPeriod = 3; // in seconds
	int nbCorr = maxPeriod / timePerFrame;
	int nbFrames = spectrogram.size().width;
	int w = nbFrames - nbCorr;
	cv::Mat corr(cv::Size(w, nbCorr), CV_64FC1);
	{
		std::cout << "Correlation with next frames" << std::endl;
		double* specP = (double*)spectrogram.data;
		double* corrP = (double*)corr.data;
		for (int frame = 0; frame < w; frame++) { // Similarity between each frame
			for (int f2 = 0; f2 < nbCorr; f2++) { // and another
				double dist = 0;
				for (int i = 0; i < specH; i++) { // Distance between FFTs amplitudes
					double diff = specP[i*nbFrames + frame + f2] - specP[i*nbFrames + frame];
					dist += diff*diff; // Euclidian distance
				}
				corrP[f2*w + frame] = sqrt(dist / nbCorr);
			}
		}
	}
	if (display) {
		cv::Mat dst = corr;
		cv::pow(dst, 0.3, dst);
		cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
		cv::imwrite(dispFolder + "corr.png", colorMap(dst));

		cv::GaussianBlur(dst, dst, cv::Size(129, 1), 100);
		cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
		cv::imwrite(dispFolder + "corrBlurred.png", colorMap(dst));
	}

	// Averaging on the whole music (TODO : doesn't work if the rythm changes)
	std::vector<double> values(nbCorr);
	{
		std::cout << "Averaging" << std::endl;
		double* corrP = (double*)corr.data;
		for (int i = 0; i < nbCorr; i++) {
			double sum = 0;
			for (int j = 0; j < w; j++) {
				sum += pow(corrP[i*w + j],1); // TODO : SSD ?
			}
			values[i] = sqrt(sum / w);
		}
		double minV = INFINITY, maxV = 0;
		for (int i = 1; i < values.size(); i++) {
			double d = values[i];
			minV = min<double>(d, minV);
			maxV = max<double>(d, maxV);
		}
		minV *= 0.99;
		for (double& d : values) { d = (d - minV) / (maxV - minV); }
	}

	// finding the period TODO : harmonics
	double period;
	{
		std::cout << "Computing the period : ";
		//double minPeriod = 1.0; // HACK
		double minV = INFINITY;
		for (int i = 0; i < values.size(); i++) {
			double v = values[i];
			if (v < minV) {
				double per = i * timePerFrame;
				if (per > minPeriod) {
					minV = v;
					period = per;
				}
			}
		}
		std::cout << period << " s" << std::endl;
	}

	// Plotting the period graph
	if (display) {
		int w = 1024; // display width
		int h = 512, margin = 32;
		cv::Mat plot(cv::Size(w, h), CV_8UC3); // plotting values
		plot = cv::Scalar{ 0,0,0 };
		for (int i = 1; i < values.size(); i++) {
			cv::line(plot,
			{ int(((i-1)*w)/values.size()), int(values[i-1]*(h-margin)) },
			{ int((i*w) / values.size()), int(values[i]*(h-margin)) },
			{ 255,255,255 });
		}
		// plotting the time scale
		int ticks = 12;
		for (int i = 0; i < ticks; i++) {
			int frame = (i*nbCorr) / (ticks + 1);
			std::stringstream ss;
			ss << " " << int(100 * frame * timePerFrame)/100.0 << "s";
			cv::line(plot, { (frame*w) / nbCorr, h - margin / 4 },
			{ (frame*w) / nbCorr, int(h - margin*0.7) }, { 255,255,255 });
			cv::putText(plot, ss.str(),
			{ (frame*w) / nbCorr, h - margin/4 },
				cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar{ 255, 255, 255 });
		}
		// plotting the best period found
		int bestP = period / timePerFrame,
			bestX = int((bestP * w) / values.size()),
			bestY = values[bestP] * (h - margin);
		cv::circle(plot, { bestX, bestY },
			5, { 255, 255, 0 });
		std::stringstream ss; ss << period << "s";
		cv::putText(plot, ss.str(), { bestX + 10,bestY },
			cv::FONT_HERSHEY_PLAIN, 1, {255,255,0} );
		cv::imwrite(dispFolder + "graph.png", plot);
	}

	// plotting the ticks on the graph
	if (display) {
		std::cout << "Exporting the period frames" << std::endl;
		int fftS = 512;
		cv::Mat spec = 0*spect(samples, fftS);
		cv::pow(spec, 0.3, spec);
		spec = colorMap(spec);
		double ratio = period * sampleRate / fftS;

		// displaying the offset as a video
		int w = ratio, h = spec.size().height;
		cv::VideoWriter video("../../data/BPM/ticks.avi",
			cv::VideoWriter::fourcc('I', 'Y', 'U', 'V'), 30, cv::Size(w,h));

		// adding bars to it
		for (int i = 0; ((i+1)*ratio) <= spec.size().width; i ++) {
			int x = i*ratio;
			cv::Mat frame = spec(cv::Rect(x, 0, w, h));
			cv::resize(frame, frame, cv::Size(w,h));
			video << frame;
			for (int y = 0; y < spec.size().height; y++) {
				spec.data[3 * (y*spec.size().width + x) + 0] = 150;
				spec.data[3 * (y*spec.size().width + x) + 1] = 100;
				spec.data[3 * (y*spec.size().width + x) + 2] = 0;
			}
		}
		cv::imwrite("../../data/BPM/ticks.png", spec);
		video.release();
	}

	// Correcting the error
	{
		// TODO
	}


	// Finding the song structure
	{
		std::cout << "Computing the song repetitions" << std::endl;

		// retrieving each frame (i.e. period)
		double ratio = period * sampleRate / fftSize;
		int w = ratio, h = spectrogram.size().height;
		std::vector<std::vector<double>> frames; // frames (one for each period)
		for (int i = 0; ((i + 1)*ratio) <= spectrogram.size().width; i++) {
			cv::Mat frame;
			spectrogram(cv::Rect(i*ratio, 0, ratio, spectrogram.size().height))
				.copyTo(frame);
			//cv::resize(frame, frame, cv::Size(w, h), 0, 0, cv::INTER_AREA); // HACK
			std::vector<double> values(w*h);
			double* frameP = (double*) frame.data;
			std::copy(frameP, frameP + w*h, values.data());
			frames.push_back(values);
		}

		// comparing each fram to the other
		int n = frames.size();
		cv::Mat corres(cv::Size(n, n), CV_64FC1);
		double* corrP = (double*)corres.data;
		for (int i = 0; i < n; i++) {
			auto& vec1 = frames[i];
			for (int j = i + 1; j < n; j++) {
				double sum = 0;
				auto& vec2 = frames[j];
				for (int k = 0; k < w*h; k++) {
					double diff = vec1[k] - vec2[k];
					sum += diff*diff;
				}
				corrP[i*n + j] = sqrt(sum / (w*h));
				corrP[j*n + i] = sqrt(sum / (w*h));
			}
		}
		//cv::pow(corres, 0.4, corres);
		cv::normalize(corres,corres, 0, 1, cv::NORM_MINMAX);
		cv::imwrite(dispFolder + "songStructure.png", colorMap(corres));
	}

	return period;
}

std::vector<double> normalize(const std::vector<double>& src) {
	double maxV = 0;
	for (double d : src) { maxV = max(maxV, abs(d)); }
	if (maxV == 0) { std::cerr << "error : null vector"
		<< std::endl; return src; }
	std::vector<double> dst = src;
	for (double& d : dst) { d = abs(d) / maxV; }
	return dst;
}

std::vector<double> descriptor(const Audio& src) {

	std::vector<double> dst;

	{
		// high precision in frequencies
		cv::Mat spectrogram = spect(src.samples, 2 * 4096);
		double* data = (double*)spectrogram.data;
		int w = spectrogram.size().width, h = spectrogram.size().height;
		std::vector<double> values(h);
		for (int y = 0; y < h; y++) {
			double sum = 0;
			for (int x = 0; x < w; x++) {
				sum += data[y*w + x];
			}
			values[y] = sum / w;
		}
		return normalize(values);
	}

	/*{
		// high temporal precison
		cv::Mat spectrogram = spect(src.samples, 4*512);
		cv::Mat edgeX, edgeY;
		cv::Sobel(spectrogram, edgeX, CV_64F, 1, 0);
		cv::Sobel(spectrogram, edgeY, CV_64F, 0, 1);
		cv::imwrite("../../data/dst.png", 128 + 128 * edgeY);
	}*/
	return dst;
};

#endif
