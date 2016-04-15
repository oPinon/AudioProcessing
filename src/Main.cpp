
#include <iostream>
#include "Audio.h"
#include "Processing.h"

int main() {

	vocoder(
		Audio("../../data/src.mp3"),
		Audio("../../data/sawtooth.mp3")
	).write("../../data/dst.mp3");
}