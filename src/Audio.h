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
	// HACK : change type
	void addFrame(const AVFrame* frame) {

		typedef int16_t type;
		const type* data = (type*)frame->data[0];
		for (int i = 0; i < frame->nb_samples/2; i++) {
			samples.push_back(double(data[i]) / INT16_MAX);
		}
	}

public :

	std::vector<double> samples;
	uint sampleRate;

	// reading an audio file
	// based on http://www.gamedev.net/topic/624876-how-to-read-an-audio-file-with-ffmpeg-in-c/?view=findpost&p=4940299
	Audio(char* fileName) {

		av_register_all(); // TODO : only once ?

		// finding the format
		AVFormatContext* formatContext = nullptr;
		if (avformat_open_input(&formatContext, fileName, nullptr, nullptr) != 0) {
			std::cerr << "Error opening " << fileName << std::endl; throw 1;
		}

		// finding the stream info
		if (avformat_find_stream_info(formatContext, nullptr) < 0) {
			avformat_close_input(&formatContext);
			std::cerr << "error finding the stream info" << std::endl; throw 1;
		}

		// finding the audio stream
		AVCodec* cdc = nullptr;
		int streamIndex = av_find_best_stream(formatContext, AVMEDIA_TYPE_AUDIO, -1, -1, &cdc, 0);
		if (streamIndex < 0) {
			std::cerr <<"could not find any stream" << std::endl;
			avformat_close_input(&formatContext);
			throw 1;
		}

		AVStream* audioStream = formatContext->streams[streamIndex];
		AVCodecContext* codecContext = audioStream->codec;
		codecContext->codec = cdc;

		// opening the codec
		if (avcodec_open2(codecContext, codecContext->codec, NULL) != 0) {
			avformat_close_input(&formatContext);
			std::cerr << "Couldn't open codec" << std::endl;
			throw 1;
		}

		// getting the sample rate
		sampleRate = codecContext->sample_rate;

		std::cout << "channels : " << codecContext->channels << std::endl;
		std::cout << "sample format : " << av_get_sample_fmt_name(codecContext->sample_fmt) << std::endl;
		std::cout << "is the format planar ? " << (av_sample_fmt_is_planar(codecContext->sample_fmt) ? "True" : "False") << std::endl;
	
		AVPacket packet;
		av_init_packet(&packet);

		AVFrame* frame = av_frame_alloc();
		if (!frame) { std::cerr << "Error allocating frame" << std::endl; throw 1; }

		// Reading packets
		while (av_read_frame(formatContext, &packet) == 0) {

			if (packet.stream_index == audioStream->index) {

				// Audio packets can have multiple audio frames in a single packet
				while (packet.size > 0) {

					// Try to decode the packet into a frame
					// Some frames rely on multiple packets, so we have to make sure the frame is finished before
					// we can use it
					int gotFrame = 0;
					int result = avcodec_decode_audio4(codecContext, frame, &gotFrame, &packet);

					if (result >= 0 && gotFrame) {
						packet.size -= result;
						packet.data += result;

						// We now have a fully decoded audio frame
						// append it to the audio data
						addFrame(frame);
					}
					else {
						packet.size = 0;
						packet.data = nullptr;
					}
				}
			}

			av_free_packet(&packet);
		}

		// Some codecs will cause frames to be buffered up in the decoding process. If the CODEC_CAP_DELAY flag
		// is set, there can be buffered up frames that need to be flushed, so we'll do that
		if (codecContext->codec->capabilities & CODEC_CAP_DELAY)
		{
			av_init_packet(&packet);
			// Decode all the remaining frames in the buffer, until the end is reached
			int gotFrame = 0;
			while (avcodec_decode_audio4(codecContext, frame, &gotFrame, &packet) >= 0 && gotFrame)
			{
				// We now have a fully decoded audio frame
				addFrame(frame);
			}
		}

		// Clean up!
		av_free(frame);
		avcodec_close(codecContext);
		avformat_close_input(&formatContext);
	}
};