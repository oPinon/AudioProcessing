
#include <iostream>
#include "Audio.h"
#include "STFT.h"

#include "kiss_fft.h"

struct LowPass : STFT<double> {

	LowPass(int fftSize = 512) : STFT<double>(fftSize) {};

	std::vector<double> process(const std::vector<double>& src) {
		
		int n = src.size();

		// creating coeffficients
		kiss_fft_cfg cfg = kiss_fft_alloc(n, false, 0, 0);
		kiss_fft_cfg cfgI = kiss_fft_alloc(n, true, 0, 0);

		std::vector<kiss_fft_cpx> input(n), output(n);
		for (int i = 0; i < n; i++) {
			input[i] = { (kiss_fft_scalar)src[i] , 0 };
		}

		// forward FFT
		kiss_fft(cfg, input.data(), output.data());

		// filtering the frequencies
		for (int f = 0; f < 0.99*n / 2; f++) {
			output[n/2-f] = { 0, 0 };
			output[n/2+f] = { 0, 0 };
		}

		// backward FFT
		kiss_fft(cfgI, output.data(), input.data());

		std::vector<double> dst(n);
		for (int i = 0; i < n; i++) {
			dst[i] = input[i].r / src.size();
		}

		// freeing coefficients
		kiss_fft_free(cfg);
		kiss_fft_free(cfgI);

		return dst;
	}
};

void PrintHelp()
{
	std::cout << "Command line arguments are : " << std::endl;
	std::cout << "-i <fileName> : (audio or image) src data" << std::endl;
	std::cout << "-o <fileName> : (audio or image) dst data" << std::endl;
	std::cout << "-v : enables verbose mode" << std::endl;
}

int main(int argc, char* argv[]) {

	bool printHelp = false;
	bool verbose = true;
	const char* inputFile = NULL;
	const char* outputFile = NULL;
	if (argc == 1)
		printHelp = true;

	for (int i = 1; i < argc; i++)
	{
		if ( _strcmpi( argv[i], "-i" ) == 0 && i + 1 < argc)
			inputFile = argv[++i];
		else
		if( _strcmpi(argv[i], "-o" ) == 0 && i + 1 < argc )
			outputFile = argv[++i];
		if( _strcmpi( argv[i], "-v" ) == 0 )
			verbose = true;
	}
	if (inputFile == NULL)
	{
		std::cerr << "missing input file" << std::endl;
		printHelp = true;
	}
	if (outputFile == NULL)
	{
		std::cerr << "missing output file" << std::endl;
		printHelp = true;
	}

	if (printHelp)
		PrintHelp();
	else
	{
		Audio src(inputFile);
		if( verbose )
			std::cout << "read " << inputFile << "; filtering.." << std::endl;
		/*LowPass stft(4096);
		Audio dst;
		int step = 47;
		for (int i = 0; i <= src.samples.size() - step; i += step) {
			stft.addSamples(src.samples.data() + i, step);
			auto out = stft.getSamples(step);
			for (int j = 0; j < out.size(); j++) {
				dst.samples.push_back(out[j]);
			}
		}*/
		Audio dst = src;
		if( outputFile )
			std::cout << "done. Writing " << outputFile << std::endl;
		dst.write(outputFile);
	}
	return 0;
}
