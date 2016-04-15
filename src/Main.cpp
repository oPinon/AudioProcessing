
#include <iostream>
#include "Audio.h"
#include "Processing.h"

int main() {

	vocoder(
		Audio("../../data/voice.mp3"),
		Audio("../../data/sawtooth.mp3"),
		256
	).write("../../data/dst.mp3");
}