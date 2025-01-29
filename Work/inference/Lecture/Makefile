#
# Makefile.c
#

TARGET = all
C = gcc
CFLAGS = -std=c99 -Wall

#Version materielle
INCLDIRS =
LIBDIRS =

LIBS64 = -lm

SRCS = $(wildcard *.c)


OBJS = ${SRCS:.c=.o}

$(TARGET): $(OBJS)
	$(C) $(CFLAGS) $(INCLDIRS) -o $@ $(OBJS) $(LIBS64)

$(OBJS):
	$(C) $(CFLAGS) $(INCLDIRS) -c $*.c


clean:
	rm -f $(OBJS) core

veryclean: clean
	rm -f $(TARGET) a.out *.*~
	rm -rf $(TARGET).dSYM
