#pragma once

// FFMPEG
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
};

#include <vector>
#include <assert.h>
#include <fstream>

typedef unsigned int uint;

// TODO : multichannel
class Audio {

	// append a frame to the samples
	void addFrame(const AVFrame* frame, AVSampleFormat format) {

		switch (format) {
		case AV_SAMPLE_FMT_S16 :
		case AV_SAMPLE_FMT_S16P: {
			const uint16_t* data = (uint16_t*)frame->data[0];
			for (int i = 0; i < frame->nb_samples; i++) {
				samples.push_back(double(data[i]) / INT16_MAX);
			}
			break;
		}
		case AV_SAMPLE_FMT_FLT :
		case AV_SAMPLE_FMT_FLTP: {
			const float* data = (float*)frame->data[0];
			for (int i = 0; i < frame->nb_samples; i++) {
				samples.push_back(double(data[i]) / 1);
			}
			break;
		}
		}
	}

public :

	std::vector<double> samples;
	uint sampleRate;

	Audio() : sampleRate(44100) {}

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
						addFrame(frame, codecContext->sample_fmt);
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
				addFrame(frame, codecContext->sample_fmt);
			}
		}

		// Clean up!
		av_free(frame);
		avcodec_close(codecContext);
		avformat_close_input(&formatContext);
	}

	void write(const char *filename)
	{
		av_register_all();

		AVCodec *codec;
		AVCodecContext *c = NULL;
		AVFrame *frame;
		AVPacket pkt;
		int ret, got_output;
		int buffer_size;
		uint16_t *samplesBuff;

		codec = avcodec_find_encoder(AV_CODEC_ID_MP3);
		if (!codec) {
			fprintf(stderr, "Codec not found\n");
			return;
		}
		c = avcodec_alloc_context3(codec);
		if (!c) {
			fprintf(stderr, "Could not allocate audio codec context\n");
			return;
		}
		
		// Encoding parameters
		c->bit_rate = pow(2,18);
		c->sample_fmt = AV_SAMPLE_FMT_S16P;
		c->sample_rate = sampleRate;
		c->channel_layout = AV_CH_LAYOUT_MONO;
		c->channels = 1;

		// Opening the codec
		if (avcodec_open2(c, codec, NULL) < 0) {
			fprintf(stderr, "Could not open codec\n");
			return;
		}

		std::fstream outFile(filename, std::ios::out | std::ios::binary);
		if (!outFile.is_open()) {
			std::cerr << "can't write " << filename << std::endl;
			return;
		}

		/* frame containing input raw audio */
		frame = av_frame_alloc();
		if (!frame) {
			fprintf(stderr, "Could not allocate audio frame\n");
			return;
		}
		frame->nb_samples = c->frame_size;
		frame->format = c->sample_fmt;
		frame->channel_layout = c->channel_layout;

		// the codec gives us the frame size, in samples,
		// we calculate the size of the samples buffer in bytes
		buffer_size = av_samples_get_buffer_size(NULL, c->channels, c->frame_size,
			c->sample_fmt, 0);
		if (buffer_size < 0) {
			fprintf(stderr, "Could not get sample buffer size\n");
			return;
		}
		std::vector<char> buffer(buffer_size);
		samplesBuff = (uint16_t*)buffer.data();
		if (!samplesBuff) {
			fprintf(stderr, "Could not allocate %d bytes for samples buffer\n",
				buffer_size);
			return;
		}
		
		// setup the data pointers in the AVFrame
		ret = avcodec_fill_audio_frame(frame, c->channels, c->sample_fmt,
			(const uint8_t*)samplesBuff, buffer_size, 0);
		if (ret < 0) {
			fprintf(stderr, "Could not setup audio frame\n");
			return;
		}

		for (int i = 0; i <= samples.size() - c->frame_size; i+= c->frame_size ) {
			av_init_packet(&pkt);
			pkt.data = NULL; // packet data will be allocated by the encoder
			pkt.size = 0;
			for (int j = 0; j < c->frame_size; j++) {
				samplesBuff[j] = (int)(samples[i+j] * INT16_MAX);
			}
			// encode the samples
			ret = avcodec_encode_audio2(c, &pkt, frame, &got_output);
			if (ret < 0) {
				fprintf(stderr, "Error encoding audio frame\n");
				exit(1);
			}
			if (got_output) {
				outFile.write((char*)pkt.data, pkt.size);
				av_packet_unref(&pkt);
			}
		}
		// get the delayed frames
		for (got_output = 1; got_output; ) {
			ret = avcodec_encode_audio2(c, &pkt, NULL, &got_output);
			if (ret < 0) {
				fprintf(stderr, "Error encoding frame\n");
				exit(1);
			}
			if (got_output) {
				outFile.write((char*)pkt.data, pkt.size);
				av_packet_unref(&pkt);
			}
		}
		outFile.close();
		av_frame_free(&frame);
		avcodec_close(c);
		av_free(c);
	}
};