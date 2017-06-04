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

template<typename T>
class STFT {

	struct Sample {
		T value; // value of the sample
		double coeff; // coefficient to normalize with
		T getValue() { return value / coeff; }
	};

	int fftSize;
public: // HACK
	CircularBuffer<T> input; // samples to be processed by FFTs
	CircularBuffer<Sample> output; // processed samples
	void computeFFT(); // computes overlapping FFTs from/to the 2 buffers
	double window(double x); // window coefficient for x in [-1;1]
	virtual std::vector<T> process(const std::vector<T>& src); // process an FFT

public:
	STFT(int fftSize = 512,
		int bufferSize = 0 // equals to 2*fftSize by default
	);
	void addSamples(const T* samples, int nbSamples);
	std::vector<T> getSamples(int nbSamples);
	std::vector<T> getSamples() { return getSamples(this->output.size()); }
};
