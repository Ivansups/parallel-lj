#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "system.hpp"

namespace md {

// Radial distribution function histogram, all pairs with minimum image.
// rmax must not exceed box/2. g() averages over all accumulated samples.
class Rdf {
public:
    Rdf(int nbins, double rmax)
        : rmax_(rmax), dr_(rmax / nbins), counts_(static_cast<std::size_t>(nbins), 0.0) {}

    void accumulate(const System& s) {
        const double rmax2 = rmax_ * rmax_;
        for (std::size_t i = 0; i < s.n; ++i)
            for (std::size_t j = i + 1; j < s.n; ++j) {
                double r2 = 0.0;
                for (int k = 0; k < 3; ++k) {
                    const double d = min_image(s.pos[3 * i + k] - s.pos[3 * j + k], s.box);
                    r2 += d * d;
                }
                if (r2 >= rmax2) continue;
                const auto bin = static_cast<std::size_t>(std::sqrt(r2) / dr_);
                counts_[std::min(bin, counts_.size() - 1)] += 2.0;  // both i and j
            }
        ++samples_;
        n_ = static_cast<double>(s.n);
        density_ = n_ / (s.box * s.box * s.box);
    }

    double dr() const { return dr_; }
    double r(int bin) const { return (bin + 0.5) * dr_; }

    std::vector<double> g() const {
        std::vector<double> out(counts_.size(), 0.0);
        if (samples_ == 0) return out;
        constexpr double kFourThirdsPi = 4.0 / 3.0 * 3.14159265358979323846;
        for (std::size_t k = 0; k < counts_.size(); ++k) {
            const double lo = static_cast<double>(k) * dr_, hi = lo + dr_;
            const double shell = kFourThirdsPi * (hi * hi * hi - lo * lo * lo);
            out[k] = counts_[k] / (static_cast<double>(samples_) * n_ * density_ * shell);
        }
        return out;
    }

private:
    double rmax_;
    double dr_;
    std::vector<double> counts_;
    int samples_ = 0;
    double n_ = 0.0;
    double density_ = 0.0;
};

}  // namespace md
