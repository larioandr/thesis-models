#ifndef STATISTICS_H_
#define STATISTICS_H_

#include <vector>
#include <string>
#include <cmath>
#include "Base.h"


/**
 * Get unbiased variance estimation.
 *
 * If no number of samples provided, returns biased estimation.
 * Otherwise, multiplies biased estimation `m2 - m1*m1` by `n/(n-1)`
 * to get unbiased estimation.
 *
 * @param m1 - sample mean
 * @param m2 - sample moment of order 2
 * @param n - number of samples
 * @return `(m2 - m1^2) * (n/(n-1))`
 */
double getUnbiasedVariance(double m1, double m2, ssize_t n = 0);


/**
 * Class representing samples series moments estimation using
 */
class Series {
public:
    Series(ssize_t nMoments, ssize_t windowSize);
    virtual ~Series();

    /**
     * Estimate new k-th moment value from the previous estimation and new samples.
     *
     * @param order - moment order (greater or equal then 1)
     * @param value - previous estimation
     * @param window - array of new samples
     * @param windowSize - number of samples to be taken from window
     * @param nRecords - total number of samples, incl. those in the window
     * @return new moment estimation
     */
    static double estimate_moment(
            int order,
            double value,
            const std::vector<double>& window,
            ssize_t windowSize,
            ssize_t nRecords
    );

    /**
     * Record new sample.
     *
     * The sample will be written into the window. If the window is full, then
     * new moments values will be estimated using `commit()`.
     *
     * @param x
     */
    void record(double x);

    /**
     * Estimate new moments values and reset sliding window.
     */
    void commit();

    /**
     * Get estimated moments values.
     */
    inline const std::vector<double>& getMoments() const {
        return moments;
    }

    /**
     * Get moment of the given order.
     */
    inline double getMoment(int order) const {
        if (order <= 0 || order > moments.size()) {
            throw std::out_of_range("illegal order");
        }
        return moments[order - 1];
    }

    inline double getMean() const { return moments[0]; }

    /**
     * Get unbiased variance.
     */
    inline double getVariance() const {
        return getUnbiasedVariance(moments[0], moments[1], nCommittedRecords);
    }

    /**
     * Get standard deviation.
     */
    inline double getStdDev() const {
        return std::pow(getVariance(), 0.5);
    }

    /**
     * Get number of recorded samples.
     */
    inline ssize_t getNumSamples() const {
        return nRecords;
    }

    /**
     * Get string representation of the Series object.
     */
    std::string toString() const;

private:
    std::vector<double> moments;
    std::vector<double> window;
    ssize_t wPos;
    ssize_t nRecords;
    ssize_t nCommittedRecords;
};


/**
 * Size distribution given with a probability mass function of values 0, 1, ..., N-1.
 */
class SizeDist {
public:
    /**
     * Create size distribution from a given PMF.
     *
     * @param pmf a vector with sum of elements equal 1.0, all elements should be non-negative.
     */
    SizeDist();
    explicit SizeDist(const std::vector<double>& pmf);
    SizeDist(const SizeDist& other);

    virtual ~SizeDist();

    /**
     * Get k-th moment of the distribution.
     *
     * @param order - number of moment
     * @return sum of i^k * pmf[i] over all i
     */
    double getMoment(int order) const;

    /**
     * Get mean value.
     */
    double getMean() const;

    /**
     * Get variance.
     */
    double getVariance() const;

    /**
     * Get standard deviation.
     */
    double getStdDev() const;

    /**
     * Get probability mass function.
     */
    inline const std::vector<double>& getPmf() const {
        return pmf;
    }

    /**
     * Get string representation.
     */
    std::string toString() const;
private:
    std::vector<double> pmf;
};


/**
 * Class for recording time-size series, e.g. system or queue size.
 *
 * Size varies in time, so here we store how long each size value
 * was kept. When estimating moments, we just divide all the time
 * on the total time and so get the probability mass function.
 */
class TimeSizeSeries {
public:
    explicit TimeSizeSeries(double time = 0.0, ssize_t value = 0);
    virtual ~TimeSizeSeries();

    /**
     * Record new value update.
     *
     * Here we record information about _previous_ value, and that
     * it was kept for `(time - prevRecordTime)` interval.
     * We also store the new value as `currValue`, so the next
     * time this method is called, information about this value
     * will be recorded.
     *
     * @param time - current time
     * @param value - new value
     */
    void record(double time, double value);

    /**
     * Estimate probability mass function.
     */
    std::vector<double> getPmf() const;

    /**
     * Estimate size distribution.
     */
    SizeDist getSizeDist() const;

    /**
     * Get string representation.
     */
    std::string toString() const;
private:
    double initTime;
    ssize_t currValue;
    double prevRecordTime;
    std::vector<double> durations;
};


struct VarData {
    double avg = 0.0;
    double std = 0.0;
    double var = 0.0;
    unsigned count = 0;
    std::vector<double> moments;

    VarData();
    VarData(const VarData& other);
    explicit VarData(const Series& series);

    std::string toString() const;
};


VarData buildVarData(const Series& series);

#endif
