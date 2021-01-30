/**
 * @author Andrey Larionov
 */
#include "Statistics.h"
#include <cmath>
#include <sstream>


namespace cqumo {

double getUnbiasedVariance(double m1, double m2, unsigned n) {
    if (n > 1) {
        auto _n = static_cast<double>(n);
        return (m2 - m1 * m1) * (_n / (_n - 1));
    }
    return m2 - m1 * m1;
}


// Class Series
// --------------------------------------------------------------------------

Series::Series(unsigned nMoments, unsigned windowSize) {
    moments.resize(nMoments, 0.0);
    window.resize(windowSize, 0.0);
    wPos = 0;
    nRecords = 0;
    nCommittedRecords = 0;
}

void Series::record(double x) {
    window[wPos++] = x;
    nRecords++;
    if (wPos >= window.size()) {
        commit();
    }
}

void Series::commit() {
    int numMoments = static_cast<int>(moments.size());
    for (int i = 0; i < numMoments; ++i) {
        moments[i] = estimate_moment(
                i + 1,
                moments[i],
                window, wPos, nRecords);
    }
    nCommittedRecords = nRecords;
    wPos = 0;
}

std::string Series::toString() const {
    std::stringstream ss;
    ss << "(Series: moments=[";
    std::copy(moments.begin(), moments.end(),
              std::ostream_iterator<double>(ss, " "));
    ss << "], nRecords=" << nRecords << ")";
    return ss.str();
}

double Series::estimate_moment(
        int order,
        double value,
        const std::vector<double> &window,
        unsigned windowSize,
        unsigned nRecords) {
    if (nRecords <= 0) {
        return value;
    }
    double accum = 0.0;
    windowSize = std::min(static_cast<unsigned>(window.size()), windowSize);
    for (unsigned i = 0; i < windowSize; ++i) {
        accum += std::pow(window[i], order);
    }
    return value * (1.0 - static_cast<double>(windowSize) / nRecords) +
           accum / nRecords;
}


// Class SizeDist
// --------------------------------------------------------------------------

SizeDist::SizeDist() : pmf(std::vector<double>(1, 1.0)) {}

SizeDist::SizeDist(std::vector<double> pmf) : pmf(std::move(pmf)) {}

double SizeDist::getMoment(int order) const {
    double accum = 0.0;
    for (unsigned i = 0; i < pmf.size(); ++i) {
        accum += std::pow(i, order) * pmf[i];
    }
    return accum;
}

double SizeDist::getMean() const {
    return getMoment(1);
}

double SizeDist::getVariance() const {
    return getMoment(2) - std::pow(getMoment(1), 2);
}

double SizeDist::getStdDev() const {
    return std::pow(getVariance(), 0.5);
}

std::string SizeDist::toString() const {
    std::stringstream ss;
    ss << "(SizeDist: mean=" << getMean() << ", std=" << getStdDev()
       << ", pmf=[";
    std::copy(pmf.begin(), pmf.end(), std::ostream_iterator<double>(ss, " "));
    ss << "])";
    return ss.str();
}


// Class TimeSizeSeries
// --------------------------------------------------------------------------

TimeSizeSeries::TimeSizeSeries(double time, unsigned value)
        : initTime(time), currValue(value), prevRecordTime(0.0) {
    durations.resize(1, 0.0);
}

TimeSizeSeries::~TimeSizeSeries() = default;

void TimeSizeSeries::record(double time, unsigned value) {
    if (durations.size() <= currValue) {
        durations.resize(currValue + 1, 0.0);
    }
    durations[currValue] += time - prevRecordTime;
    prevRecordTime = time;
    currValue = value;
}

std::vector<double> TimeSizeSeries::getPmf() const {
    std::vector<double> pmf(durations);
    double dt = prevRecordTime - initTime;
    for (unsigned i = 0; i < pmf.size(); ++i) {
        pmf[i] /= dt;
    }
    return pmf;
}

SizeDist TimeSizeSeries::getSizeDist() const {
    return SizeDist(getPmf());
}

std::string TimeSizeSeries::toString() const {
    std::stringstream ss;
    ss << "(TimeSizeSeries: durations=[";
    std::copy(durations.begin(), durations.end(),
              std::ostream_iterator<double>(ss, " "));
    ss << "])";
    return ss.str();
}


// Class VarData
// --------------------------------------------------------------------------

// VarData::VarData(const VarData &other)
//        : avg(other.avg),
//          std(other.std),
//          var(other.var),
//          count(other.count),
//          moments(other.moments) {}

VarData::VarData(const Series &series)
        : avg(series.getMean()),
          std(series.getStdDev()),
          var(series.getVariance()),
          count(series.getNumSamples()),
          moments(series.getMoments()) {}

std::string VarData::toString() const {
    std::stringstream ss;
    ss << "(VarData: avg=" << avg
       << ", var=" << var
       << ", std=" << std
       << ", count=" << count
       << ", moments=[" << cqumo::toString(moments) << "])";
    return ss.str();
}


// Class Counter
// --------------------------------------------------------------------------

Counter::Counter(int initValue) : value_(initValue) {}

Counter &Counter::operator=(const Counter &rside) {
    value_ = rside.get();
    return *this;
}

std::string Counter::toString() const {
    std::stringstream ss;
    ss << "(Counter: value=" << value_ << ")";
    return ss.str();
}


}
