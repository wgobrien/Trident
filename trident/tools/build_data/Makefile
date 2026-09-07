CXX := g++
CXXFLAGS := -O3 -std=c++17
LDFLAGS := -lz

# Auto-detect OpenMP support
OPENMP_TEST := $(shell echo 'int main(){}' | $(CXX) -fopenmp -x c++ - -o /dev/null 2>/dev/null && echo yes)
ifeq ($(OPENMP_TEST),yes)
  CXXFLAGS += -fopenmp
  LDFLAGS += -fopenmp
endif

COVERAGE_TARGET := coverage
COVERAGE_OBJECTS := coverage.o mmscanner.o

COUNTS_TARGET := build_counts
COUNTS_OBJECTS := build_counts.o mmscanner.o

ALL_TARGETS := $(COVERAGE_TARGET) $(COUNTS_TARGET)
ALL_OBJECTS := coverage.o build_counts.o mmscanner.o accessions.o

READS :=
PAIR :=
LIBRARY :=
TAXIDS :=

all: $(ALL_TARGETS)

$(COVERAGE_TARGET): $(COVERAGE_OBJECTS)
	$(CXX) $(CXXFLAGS) $(COVERAGE_OBJECTS) $(LDFLAGS) -o $@

$(COUNTS_TARGET): $(COUNTS_OBJECTS)
	$(CXX) $(CXXFLAGS) $(COUNTS_OBJECTS) $(LDFLAGS) -o $@

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(ALL_TARGETS) $(ALL_OBJECTS)

test-coverage: $(COVERAGE_TARGET)
	./$(COVERAGE_TARGET) $(READS) $(LIBRARY) out.tsv --pair $(PAIR) --threads 32

test-counts: $(COUNTS_TARGET)
	./$(COUNTS_TARGET) $(READS) $(LIBRARY) $(TAXIDS) --pair $(PAIR) --output counts.tsv --threads 32

valgrind: CXXFLAGS := -O0 -g -std=c++17 $(if $(filter yes,$(OPENMP_TEST)),-fopenmp)
valgrind: clean $(COVERAGE_TARGET)
	valgrind \
	  --tool=memcheck \
	  --leak-check=full \
	  --show-leak-kinds=all \
	  --track-origins=yes \
	  --error-exitcode=1 \
	  ./$(COVERAGE_TARGET) $(READS) $(LIBRARY) out.tsv --pair $(PAIR) --threads 32

.PHONY: all clean test-coverage test-counts valgrind
