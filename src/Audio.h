#pragma once

// FFMPEG
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
};

#include <vector>

typedef unsigned int uint;

// TODO : multichannel
class Audio {

	// append a frame to the samples
	void addFrame(const AVFrame* frame, AVSampleFormat format);

public :

	std::vector<double> samples;
	uint sampleRate;

	Audio() : sampleRate(44100) {}

	// reading an audio file
	// based on http://www.gamedev.net/topic/624876-how-to-read-an-audio-file-with-ffmpeg-in-c/?view=findpost&p=4940299
	Audio(const char* fileName);

	void writeAudio(const char *fileName, AVCodecID) const;
	void write(const char *filename) const;

	void spectrogram(const char* fileName, int fftSize = 512) const;

	double period(double minPeriod = 0.2, double maxPeriod = 3) const;
};
