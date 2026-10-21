#pragma once

#include <vector>

#include "system.hpp"

namespace md {

// Mean squared displacement from the construction time. Unwraps periodic
// positions, so update() must be called often enough that no particle moves
// more than box/2 between calls.
class MsdTracker {
public:
    explicit MsdTracker(const System& s) : origin_(s.pos), unwrapped_(s.pos), last_(s.pos) {}

    void update(const System& s) {
        for (std::size_t i = 0; i < unwrapped_.size(); ++i) {
            unwrapped_[i] += min_image(s.pos[i] - last_[i], s.box);
            last_[i] = s.pos[i];
        }
    }

    double msd() const {
        double sum = 0.0;
        for (std::size_t i = 0; i < unwrapped_.size(); ++i) {
            const double d = unwrapped_[i] - origin_[i];
            sum += d * d;
        }
        return sum / static_cast<double>(unwrapped_.size() / 3);
    }

private:
    std::vector<double> origin_, unwrapped_, last_;
};

}  // namespace md
