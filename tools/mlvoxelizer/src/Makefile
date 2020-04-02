CC= gcc
FFTW_INC=/usr/include/
FFTW_DIR=/usr/lib/x86_64-linux-gnu/
OPTS= -I$(FFTW_INC) #-std=gnu99

# Debug options
GCFLAGS= -g -Wall
GLIBS=   -fopenmp -lm -lpthread -lGL -lGLU -lX11 -L $(FFTW_DIR) -lfftw3 -lfftw3_omp

# Release options
CFLAGS= -O3 -fopenmp -Wall
LIBS=   -fopenmp -lm -lpthread -lGL -lGLU -lX11 -L $(FFTW_DIR) -lfftw3 -lfftw3_omp

# No-GUI options
NCFLAGS= -O3 -DNOGUI -fopenmp #-Wall
NLIBS=  -fopenmp -lm -lpthread -L $(FFTW_DIR) -lfftw3 -lfftw3_omp 

all: mlvoxelizer

nogui: CFLAGS = $(NCFLAGS)
nogui: LIBS   = $(NLIBS)
nogui: mlvoxelizer-nogui

debug: CFLAGS = $(GCFLAGS)
debug: LIBS   = $(GLIBS)
debug: clean mlvoxelizer

# Obj deps

util.o: util.c util.h Makefile
	${CC} ${OPTS} ${CFLAGS} -c util.c

random.o: random.c random.h Makefile
	${CC} ${OPTS} ${CFLAGS} -c random.c

vector.o: vector.c vector.h Makefile
	${CC} ${OPTS} ${CFLAGS} -c vector.c

matrix.o: matrix.c matrix.h Makefile
	${CC} ${OPTS} ${CFLAGS} -c matrix.c

io_bitmap.o: io_bitmap.c io_bitmap.h Makefile
	${CC} ${OPTS} ${CFLAGS} -c io_bitmap.c

voxelizer.o: voxelizer.c voxelizer.h Makefile
	${CC} ${OPTS} ${CFLAGS} -c voxelizer.c

chemio.o: chemio.c chemio.h Makefile
	${CC} ${OPTS} ${CFLAGS} -c chemio.c

atom.o: atom.c atom.h Makefile
	${CC} ${OPTS} ${CFLAGS} -c atom.c

fdp.o: fdp.c fdp.h Makefile
	${CC} ${OPTS} ${CFLAGS} -c fdp.c

rotation.o: rotation.c rotation.h Makefile
	${CC} ${OPTS} ${CFLAGS} -c rotation.c

# GUI deps

gui_button.o: gui_button.c gui_button.h gui.h types.h Makefile
	${CC} ${OPTS} ${CFLAGS} -c gui_button.c

gui_stats.o: gui_stats.c gui_stats.h gui.h types.h Makefile
	${CC} ${OPTS} ${CFLAGS} -c gui_stats.c

gui_3dview.o: gui_3dview.c gui_3dview.h gui.h types.h Makefile
	${CC} ${OPTS} ${CFLAGS} -c gui_3dview.c

gui.o: gui.c gui.h types.h Makefile
	${CC} ${OPTS} ${CFLAGS} -c gui.c

# Main

main.o: main.c main.h Makefile
	${CC} ${OPTS} ${CFLAGS} -c main.c

# Link

mlvoxelizer: util.o random.o vector.o matrix.o io_bitmap.o atom.o fdp.o rotation.o voxelizer.o chemio.o gui_button.o gui_stats.o gui_3dview.o gui.o main.o Makefile
	${CC} ${OPTS} -o mlvoxelizer util.o random.o vector.o matrix.o io_bitmap.o atom.o fdp.o rotation.o voxelizer.o chemio.o gui_button.o gui_stats.o gui_3dview.o gui.o main.o ${LIBS}

mlvoxelizer-nogui: util.o random.o vector.o matrix.o atom.o fdp.o rotation.o voxelizer.o chemio.o main.o Makefile
	${CC} ${OPTS} -o mlvoxelizer util.o random.o vector.o matrix.o atom.o fdp.o rotation.o voxelizer.o chemio.o main.o ${LIBS}

# Maintenance

clean:
	rm -f mlvoxelizer *.o

strip: clean
	rm -f *~ \#*
