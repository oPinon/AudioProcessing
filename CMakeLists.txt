cmake_minimum_required(VERSION 2.4)
project(AudioProcessing)

set( SrcDir ${CMAKE_CURRENT_SOURCE_DIR}/src )
set( FFMPEGDir "" CACHE PATH "Directory of the compiled FFMPEG library" )
if( FFMPEGDir STREQUAL "" )
	message( SEND_ERROR "Please specify the variable FFMPEGDir" )
endif()
set( KissFFTDir ${CMAKE_CURRENT_SOURCE_DIR}/deps/kissfft )
if( NOT EXISTS ${KissFFTDir} )
	message( SEND_ERROR "Please init the KissFFT submodule : git submodule --init" )
endif()

include_directories(
	${SrcDir}
	${FFMPEGDir}/include
	${KissFFTDir}
)

link_directories(
	${FFMPEGDir}/lib
)

add_executable(
	AudioProcessing
	${KissFFTDir}/kiss_fft.c
	${SrcDir}/Main.cpp
	${SrcDir}/Audio.h
	${SrcDir}/Audio.cpp
	${SrcDir}/STFT.h
	${SrcDir}/STFT.cpp
)

target_link_libraries(
	AudioProcessing
	avcodec.lib
	avdevice.lib
	avfilter.lib
	avformat.lib
	avutil.lib
	postproc.lib
	swresample.lib
	swscale.lib
)
