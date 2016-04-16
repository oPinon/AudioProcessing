#include "Audio.h"

Audio vocoder(const Audio& modulator, const Audio& frequencies, int bands = 16);

// Describe a music with values in [0;1]
std::vector<double> descriptor(const Audio& src);