#pragma once

#include <iostream>
#include "Audio.h"
#include "Processing.h"
#include <opencv2\opencv.hpp>
#include <unordered_map>

#include "Learning.h"

struct Entry {
	std::string fileName;
	std::vector<double> input; // descriptor
	int output; // music rating

	Entry() {}
	Entry(std::string name, std::vector<double> input, int output)
		: fileName(name), input(input), output(output) {}
	Entry(std::fstream& dbIn) {

		// reading the name
		int nameLength;
		dbIn.read((char*)&nameLength, sizeof(int));
		std::vector<char> name(nameLength);
		dbIn.read(name.data(), nameLength * sizeof(char));

		// reading the data
		int dataLength;
		dbIn.read((char*)&dataLength, sizeof(int));
		std::vector<double> data(dataLength);
		dbIn.read((char*)data.data(), dataLength * sizeof(double));

		// reading the rating (=class)
		int rating;
		dbIn.read((char*)&rating, sizeof(int));

		std::string fileName = name.data();

		this->fileName = name.data();
		this->input = data;
		this->output = rating;
	}
	void write(std::fstream& dbOut) const {

		// name
		int nameSize = fileName.size();
		dbOut.write((char*)&nameSize, sizeof(int));
		dbOut.write(fileName.data(), nameSize);
		// data
		int inputSize = input.size();
		dbOut.write((char*)&inputSize, sizeof(int));
		dbOut.write((char*)input.data(), inputSize * sizeof(double));
		// rating
		dbOut.write((char*)&output, sizeof(int));
	}
};

// TODO : UTF8 chars
void classification() {

	double magicNumber = 42.314; // to check the integrity of the db

	std::unordered_map<std::string, Entry> musics;

	// reading the database
	std::string dbPath = "../../data/music.db";
	std::fstream dbIn(dbPath, std::ios::in | std::ios::binary);
	if (dbIn.is_open()) {

		double key;
		while (!dbIn.eof()) {

			// reading the key (to check integrity)
			key = 0;
			dbIn.read((char*)&key, sizeof(double));
			if (key != magicNumber) { continue; } // HACK

												  // appending to the database
			Entry entry(dbIn);
			musics[entry.fileName] = entry;
		}
		dbIn.close();
	}
	std::cout << "Imported " << musics.size() << " entries from " << dbPath << std::endl;

#if 0
	// list of all musics
	std::string musicListPath = "../../data/Playlists/list.txt";
	std::fstream musicList(musicListPath, std::ios::in);
	assert(musicList.is_open());

	// descriptor database
	std::fstream dbOut(dbPath, std::ios::app | std::ios::binary);
	if (!dbOut.is_open()) { std::cerr << "can't write " << dbPath << std::endl; return; }

	// fileName \n rating \n etc...
	std::string line;
	while (!musicList.eof()) {

		// getting the file info
		std::getline(musicList, line);
		if (line.size() == 0) { break; }
		std::string name = line; // fileName
		std::getline(musicList, line);
		int rating = std::stoi(line); // rating

									  // if the music is not yet in the database
		if (musics.find(name) == musics.end()) {

			std::cout << "adding " << name << std::endl;

			try {
				Audio audio(name.c_str());
				std::vector<double> input = descriptor(audio);

				Entry entry = {
					name,
					input,
					rating
				};
				musics[name] = entry;

				// add it to the db
				dbOut.write((char*)&magicNumber, sizeof(double));
				entry.write(dbOut);
			}
			catch (...) {
				std::cerr << "error" << std::endl;
			}
		}
	}
	std::cout << "Total of  " << musics.size() << " entries" << std::endl;
#endif

	// formatting the descriptor
	/*for (auto& e : musics) {
	std::vector<double>& values = e.second.input;
	double var = 0;
	for (double d : values) { var += d*d; }
	var = sqrt(var / values.size());
	for (double& d : values) { d = fmin(1,fmax(0,d/var)); }
	}*/

	// Machine learning
	std::vector<Sample> samples;
	int n = musics.begin()->second.input.size();
	for (auto& e : musics) {
		Sample s;
		s.input = std::vector<double>(e.second.input.rbegin(), e.second.input.rend());
		double value = (e.second.output > 2 ? 1 : 0);
		s.output = { value };
		samples.push_back(s);
	}

	srand(time(NULL));
	random_shuffle(samples.begin(), samples.end());

#if 0
	for (Sample& s : samples) {

		std::cout << "classified as " << s.output[0] << std::endl;

		// displaying the descriptor
		int n = s.input.size();
		int h = 256;
		cv::Mat plot(cv::Size(n, h), CV_8UC1);
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < n; x++) {
				plot.data[y*n + x] = ((s.input[x] * h) < (h - y)) ? 0 : 255;
			}
		}
		cv::resize(plot, plot, cv::Size(1024, h));
		cv::imshow("descriptor", plot);
		if (cv::waitKey(16) == 27) { break; }
	}
#endif

	int nbNegative = 0;
	for (const Sample& s : samples) {
		if (s.output[0] < 0.5) { nbNegative++; }
	}
	std::cout << "percentage of negatives : "
		<< (nbNegative * 100) / samples.size() << std::endl;

	NetLearner classifier(Network({ n, 1 }, 0.0));

	// separating train and test
	int nbTrain = 0.8*samples.size();
	std::vector<Sample> trainSet(samples.begin(), samples.begin() + nbTrain);
	std::vector<Sample> testSet(samples.begin() + nbTrain, samples.end());

	while (true) {

		std::cout << std::endl;

		double error = classifier.learn(trainSet, 10, 16, 1);
		std::cout << "error on learning " << 100 * error << std::endl;

		int testErr = 0;
		for (const Sample& s : testSet) {
			double res = classifier.apply(s.input)[0];
			if ((res < 0.5) != (s.output[0] < 0.5)) { testErr++; }
		}
		std::cout << "error on test " << (testErr * 100) / testSet.size() << std::endl;

		// displaying the neural network
		/*int h = 256;
		cv::Mat plot(cv::Size(n, h), CV_8UC1);
		double* coeffsP = classifier.net.synapses[0].coefficients.data();
		for (int y = 0; y < h; y++) {
		for (int x = 0; x < n; x++) {
		double v = 0.5 + 0.2*coeffsP[x];
		plot.data[y*n + x] = ((v * h) < (h - y)) ? 0 : 255;
		}
		}
		cv::resize(plot, plot, cv::Size(1024, h));
		cv::imshow("NN coeffs", plot);
		if (cv::waitKey(16) == 27) { break; }*/
	}
}