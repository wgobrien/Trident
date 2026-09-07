SUBDIR := trident/tools/build_data

.PHONY: all clean test-coverage test-counts valgrind

all:
	$(MAKE) -C $(SUBDIR)

clean:
	$(MAKE) -C $(SUBDIR) clean

test-coverage:
	$(MAKE) -C $(SUBDIR) test-coverage

test-counts:
	$(MAKE) -C $(SUBDIR) test-counts

valgrind:
	$(MAKE) -C $(SUBDIR) valgrind
