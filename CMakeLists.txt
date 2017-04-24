cmake_minimum_required(VERSION 2.4)
project(AudioProcessing)

if(NOT WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

option( WithOpenCV "Use the features that depend on OpenCV" OFF )
set( SrcDir ${CMAKE_CURRENT_SOURCE_DIR}/src )

set( FFMPEGDir "" CACHE PATH "Directory of the compiled FFMPEG library" )
if( WIN32 )
	if( FFMPEGDir STREQUAL "" )
		message( SEND_ERROR "Please specify the variable FFMPEGDir" )
	endif()
else()
	set( FFMPEGDir "/" )
endif()

set( KissFFTDir ${CMAKE_CURRENT_SOURCE_DIR}/deps/kissfft )
if( NOT EXISTS ${KissFFTDir} )
	message( SEND_ERROR "Please init the KissFFT submodule : git submodule --init" )
endif()

if( WithOpenCV )
	set( OpenCVDir "" CACHE PATH "Directory of the compiled OpenCV library" )
	if( OpenCVDir STREQUAL "" )
		message( SEND_ERROR "Please specify the variable OpenCVDir or disable the option WithOpenCV" )
	endif()
	include_directories( ${OpenCVDir}/include )
	link_directories( ${OpenCVDir}/lib )
	add_definitions(-DAUDIOPROCESSING_USE_OPENCV)
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
	${SrcDir}/Processing.h
	${SrcDir}/Processing.cpp
)

target_link_libraries(
	AudioProcessing
	avcodec
	avdevice
	avfilter
	avformat
	avutil
	swresample
	swscale
)

if( WithOpenCV )
	target_link_libraries( AudioProcessing opencv_world310 )
endif()
