
#include <iostream>
#include "Audio.h"
#include "Processing.h"

#include <opencv2\opencv.hpp>

void main(int argc, char* argv[]) {

	if (argc < 2) {
		std::cout << "Enter an .mp3 file as input" << std::endl;
		return;
	}

	//Audio src(argv[1]);
	Audio src("../../data/voice.mp3");
	int bpm = src.bpm();

	Audio dst;
	int n = 20000;
	for (int i = 0; i < n; i++) {
		double t = double(i) / n;
		double v =  64 * sin(440 * i * 2 * 3.14 / dst.sampleRate)
			* t * exp(-30*t);
		dst.samples.push_back(v);
	}

	for (int i = 0; i < src.samples.size(); i += src.sampleRate * 60 / bpm) {
		for (int j = 0; j < dst.samples.size(); j++) {
			if (i + j >= src.samples.size()) { break; }
			src.samples[i + j] = 0.5*src.samples[i + j] +0.5*dst.samples[j];
		}
	}
	src.write("../../data/dst.mp3");
}