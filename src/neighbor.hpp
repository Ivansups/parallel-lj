#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "forces.hpp"
#include "lj.hpp"
#include "system.hpp"

namespace md {

// Verlet neighbour list built with cell lists; pairs within rc + skin.
class NeighborList {
public:
    using Pair = std::pair<std::uint32_t, std::uint32_t>;

    explicit NeighborList(double skin = 0.3) : skin_(skin) {}

    void build(const System& s) {
        pairs_.clear();
        ref_pos_ = s.pos;
        const double rl = lj::kCutoff + skin_;
        const double rl2 = rl * rl;
        const int nc = std::max(1, static_cast<int>(s.box / rl));
        const double cell = s.box / nc;

        std::vector<std::vector<std::uint32_t>> members(static_cast<std::size_t>(nc) * nc * nc);
        for (std::size_t i = 0; i < s.n; ++i) {
            int idx[3];
            for (int k = 0; k < 3; ++k)
                idx[k] = std::min(nc - 1, std::max(0, static_cast<int>(s.pos[3 * i + k] / cell)));
            members[cell_index(idx[0], idx[1], idx[2], nc)].push_back(static_cast<std::uint32_t>(i));
        }

        for (int ix = 0; ix < nc; ++ix)
            for (int iy = 0; iy < nc; ++iy)
                for (int iz = 0; iz < nc; ++iz) {
                    const std::size_t c = cell_index(ix, iy, iz, nc);
                    // With fewer than 3 cells per side, wrapped neighbours coincide.
                    std::vector<std::size_t> nbrs;
                    for (int dx = -1; dx <= 1; ++dx)
                        for (int dy = -1; dy <= 1; ++dy)
                            for (int dz = -1; dz <= 1; ++dz)
                                nbrs.push_back(cell_index(ix + dx, iy + dy, iz + dz, nc));
                    std::sort(nbrs.begin(), nbrs.end());
                    nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());

                    for (std::size_t c2 : nbrs) {
                        if (c2 < c) continue;
                        for (std::size_t a = 0; a < members[c].size(); ++a)
                            for (std::size_t b = (c2 == c ? a + 1 : 0); b < members[c2].size(); ++b)
                                if (distance2(s, members[c][a], members[c2][b]) < rl2)
                                    pairs_.emplace_back(members[c][a], members[c2][b]);
                    }
                }
    }

    // True once any particle moved more than skin/2 since the last build.
    bool needs_rebuild(const System& s) const {
        if (ref_pos_.size() != s.pos.size()) return true;
        const double limit2 = 0.25 * skin_ * skin_;
        for (std::size_t i = 0; i < s.n; ++i) {
            double d2 = 0.0;
            for (int k = 0; k < 3; ++k) {
                const double d = min_image(s.pos[3 * i + k] - ref_pos_[3 * i + k], s.box);
                d2 += d * d;
            }
            if (d2 > limit2) return true;
        }
        return false;
    }

    const std::vector<Pair>& pairs() const { return pairs_; }

private:
    static std::size_t cell_index(int x, int y, int z, int nc) {
        const auto wrap = [nc](int v) { return static_cast<std::size_t>(((v % nc) + nc) % nc); };
        return (wrap(x) * nc + wrap(y)) * nc + wrap(z);
    }

    static double distance2(const System& s, std::size_t i, std::size_t j) {
        double r2 = 0.0;
        for (int k = 0; k < 3; ++k) {
            const double d = min_image(s.pos[3 * i + k] - s.pos[3 * j + k], s.box);
            r2 += d * d;
        }
        return r2;
    }

    double skin_;
    std::vector<double> ref_pos_;
    std::vector<Pair> pairs_;
};

// Forces from the neighbour list; fills s.force and returns the potential energy.
inline double compute_forces(System& s, const NeighborList& nl) {
    const double rc2 = lj::kCutoff * lj::kCutoff;
    std::fill(s.force.begin(), s.force.end(), 0.0);
    s.virial = 0.0;
    double pe = 0.0;
    for (const auto& p : nl.pairs()) pe += add_pair_force(s, p.first, p.second, rc2);
    return pe;
}

}  // namespace md
