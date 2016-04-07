#pragma once

#include <vector>

void testSTFT();

template<typename T>
class CircularBuffer {
	std::vector<T> values; // TODO : adaptive size ?
	int _start = 0, _size = 0;
public:
	CircularBuffer(int size = 4096) : values(size) {};
	void pop(int number);
	void push_back(const T& v);
	int size();
	T& operator[](int i);
};

class STFT {

	struct Sample {
		double value, // value of the sample
			coeff; // coefficient to normalize with
		double getValue() { return value / coeff; }
	};

	int fftSize;
public: // HACK
	CircularBuffer<double> input; // samples to be processed by FFTs
	CircularBuffer<Sample> output; // processed samples
	void computeFFT(); // computes overlapping FFTs from/to the 2 buffers
	double window(double x); // window coefficient for x in [-1;1]
	virtual std::vector<double> STFT::process(const std::vector<double>& src); // process an FFT

public:
	STFT(int fftSize = 512);
	void addSamples(double* samples, int nbSamples);
	std::vector<double> getSamples(int nbSamples);
};