#CXXFLAGS = -I include  -std=c++11 -O3  -I/home/ist/anaconda3/envs/food/include/python2.7
#CXXFLAGS = -I include -std=c++11 -O3 -I /home/ist/anaconda3/include/python3.7m
CXXFLAGS = -I include -std=c++11 -O3

DEPS = lanms.h $(shell find include -xtype f)
CXX_SOURCES = adaptor.cpp include/clipper/clipper.cpp
#OPENCV = `pkg-config --cflags --libs opencv`
#OPENCV = /home/ist/anaconda3/envs/food/lib
#OPENCV = /home/ist/anaconda3/lib
#-WI,-rpath = LIB_SO = adaptor.so

INCLUDES = -I/home/ist/anaconda3/envs/food/include
LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml
LIBDIRS = -L/home/ist/anaconda3/envs/food/lib

LIB_SO = adaptor.so

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC $(OPENCV)

clean:
	rm -rf $(LIB_SO)
