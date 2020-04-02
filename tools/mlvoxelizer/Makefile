# By default build everything:
all: src/mlvoxelizer

gui:
	cd src && $(MAKE)

# Build the voxelizer if needed
src/mlvoxelizer:
	cd src && $(MAKE) nogui

# Remove produced data as well as temp files:
clean:
	cd src/ && $(MAKE) clean
	rm -f map/* vox/* lig/*
	rm -rf *.voxels
	rm -rf *.ligands
