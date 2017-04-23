#include "STFT.h"

template<typename A, typename B>
inline A max(const A& a, const B& b) { return a < b ? b : a; }
template<typename A, typename B>
inline A min(const A& a, const B& b) { return a < b ? a : b; }

//#define _DEBUG 1

template<typename T>
int CircularBuffer<T>::size() {
	return _size;
}

template<typename T>
void CircularBuffer<T>::pop(int number)
{
#ifdef _DEBUG
	if (number > this->size()) { std::cerr << "popping " << number << " out of " << this->size() << std::endl; throw 1; }
#endif
	_start = (_start + number) % values.size();
	_size -= number;
}

template<typename T>
void CircularBuffer<T>::push_back(const T& v)
{
#ifdef _DEBUG
	if (_size == values.size()) { std::cerr << "Buffer overflow (" << _size << " out of " << values.size() << ")" << std::endl; throw 1; }
#endif
	(*this)[_size] = v;
	_size++;
}

template<typename T>
inline T & CircularBuffer<T>::operator[](int i)
{
#if _DEBUG
	if (i > values.size()) { std::cerr << "accessed " << i << " out of a " << values.size() << "buffer" << std::endl; throw 1; }
#endif
	int pos = (_start + i) % values.size();
	return values[pos];
}

template<typename T>
STFT<T>::STFT(int fftSize, int bufferSize) : fftSize(fftSize) {
	bufferSize = max(bufferSize, 2*fftSize);
	input = CircularBuffer<T>(bufferSize);
	output = CircularBuffer<Sample>(bufferSize);
	for (int i = 0; i < 2*fftSize; i++) {
		output.push_back({ 0, 1 }); // TODO : std::limit
	}
}


template<typename T>
std::vector<T> STFT<T>::process(const std::vector<T>& src) {

	return src;
}

template<typename T>
double STFT<T>::window(double x) {
	return 0.5 + 0.5*cos(3.14159*x); // Hanning
}

// computing overlapping FFTs
template<typename T>
void STFT<T>::computeFFT() {

	std::vector<double> values;

	// extracting from the input
	for (int i = 0; i < fftSize; i++) { // overlaping
		double coeff = window(double(i - fftSize / 2) / (fftSize / 2));
		values.push_back(coeff * input[i]);
	}
	input.pop(fftSize/2);

	values = process(values);

	// pasting to the output
	for (int i = 0; i < fftSize; i++) {
		double coeff = window(double(i - fftSize / 2) / (fftSize / 2));
		double value = values[i] * coeff;
		if (i < fftSize/2) { // updating the sample already there (first half)
			Sample& s = output[output.size() - fftSize/2 + i];
			s.value += value;
			s.coeff += coeff*coeff; // multiplied by coeff before and after the FFT
		}
		else {
			output.push_back({ value, coeff*coeff }); // adding the new samples (second half)
		}
	}
}

template<typename T>
void STFT<T>::addSamples(T* samples, int nbSamples) {

	for (int i = 0; i < nbSamples; i++) {
		input.push_back(samples[i]);
		if (input.size() >= fftSize) { computeFFT(); }
	}
}

#include <iostream>

template<typename T>
std::vector<T> STFT<T>::getSamples(int nbSamples) {
	
	if (nbSamples > output.size()) {
		std::cerr << "reading " << nbSamples << " out of a "
			<< output.size() << " buffer" << std::endl; throw 1;
	}
	std::vector<T> dst(nbSamples);
	for (int i = 0; i < nbSamples; i++) { dst[i] = output[i].getValue(); }
	output.pop(nbSamples);
	return dst;
}

#ifdef AUDIOPROCESSING_USE_OPENCV

#include <ctime>

#include <opencv2\opencv.hpp>

void testSTFT() {

	//srand(time(NULL));
	std::cout << "Testing the STFT :" << std::endl;
	STFT<double> stft(32);

	// random input
	std::vector<double> input(512);
	input[0] = 1;
	for (int i = 1; i < input.size(); i++) {
		input[i] = 1 - 2 * double(rand()) / RAND_MAX;
	}
	
	// sending and retrieving from buffer
	std::vector<double> output;
	int step = 7;
	for (int i = 0; i < input.size(); i += step) {
		stft.addSamples(&input[i], min<int>(step, input.size() - i));
		//std::cout << stft.input.size() << std::endl;
		if (true) {//1.01 * rand() < RAND_MAX) {
			std::vector<double> samples = stft.getSamples(min<int>(step, input.size() - i));
			for (double s : samples) { output.push_back(s); }
		}
	}

	int i = 0; // position in the input signal
	std::vector<double> samples = stft.getSamples(input.size()-output.size());
	for (double s : samples) { output.push_back(s); }
	while (output[i] == 0) { i++; }
	int offset = i;

	// comparing the input and the output
	std::cout << offset << " samples of delay" << std::endl;
	samples = stft.getSamples(input.size() + offset - output.size());
	for (double s : samples) { output.push_back(s); }

	// plotting the 2 signals
	{
		int w = 1500, h = 256;
		cv::Mat inS(cv::Size(w, h), CV_8UC3);
		for (int i = 1; i < input.size(); i++) {
			cv::line(inS,
			{ int((w*(i-1)) / input.size()), int((-input[i - 1] + 1)*h / 2) },
			{ int((w*i) / input.size()), int((-input[i] + 1)*h / 2) }, { 255,255,255 });
		}
		cv::imshow("input", inS);

		cv::Mat outS(cv::Size(w, h), CV_8UC3);
		for (int i = 1; i < output.size(); i++) {
			cv::line(outS,
			{ int((w*(i - 1)) / output.size()), int((-output[i - 1] + 1)*h / 2) },
			{ int((w*i) / output.size()), int((-output[i] + 1)*h / 2) }, { 255,255,255 });
		}
		cv::imshow("output", outS);
	}

	while (i < input.size()) {
		double in = input[i - offset];
		double out = output[i];
		if (abs(in - out) > 1E-8) {
			std::cerr << "\a difference at sample " << (i - offset)
				<< " (in=" << in << " vs out=" << out << ")" << std::endl;
			cv::waitKey();
			return;
		}
		i++;
	}
	std::cout << "everything OK (" << input.size() << " samples identical)" << std::endl;
	cv::waitKey();
}
#else

// HACK : just to force template compilation
void test()
{
	STFT<double> s(32);
	s.addSamples(NULL, 0);
	s.getSamples(0);
}

#endif
