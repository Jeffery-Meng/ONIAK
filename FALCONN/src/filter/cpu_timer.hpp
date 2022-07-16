#ifndef __CPU_TIMER_HPP__
#define __CPU_TIMER_HPP__
#include <ctime>
#include <cassert>

class CPUTimer {
bool running_;
time_t start_, accu_;

public:
CPUTimer() {
    reset();
}

void reset () {
    running_ = false;
    accu_ = 0;
    start_ = 0;
}

void start() {
    assert(!running_);
    start_ = clock();
    running_ = true;
}

void stop () {
    assert(running_);
    accu_ += clock() - start_;
    running_ = false;
}

double watch() {
    assert(!running_);
    return (double) accu_ / (double) CLOCKS_PER_SEC;
}

void flip () {
    if (running_) stop();
    else start();
}

};


#endif