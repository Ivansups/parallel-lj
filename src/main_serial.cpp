// Serial LJ MD: Berendsen equilibration, then NVE. Thermo CSV on stdout;
// g(r) and MSD(t) are written to <prefix>_rdf.csv and <prefix>_msd.csv.
// Usage: md_serial cells equil_steps nve_steps seed rho T rc prefix
#include <cstdio>
#include <cstdlib>
#include <string>

#include "integrator.hpp"
#include "lattice.hpp"
#include "msd.hpp"
#include "neighbor.hpp"
#include "rdf.hpp"
#include "thermo.hpp"
#include "thermostat.hpp"

namespace {

constexpr double kDt = 0.005;
constexpr double kTau = 0.1;
constexpr int kThermoEvery = 10;
constexpr int kRdfEvery = 100;
constexpr int kRdfBins = 200;

void print(const char* phase, int step, const md::System& s, double pe) {
    const md::Thermo t = md::measure(s, pe);
    std::printf("%s,%d,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", phase, step, step * kDt, t.kinetic,
                t.potential, md::plain_potential(s, pe), t.total, t.temperature, t.pressure);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 9) {
        std::fputs("usage: md_serial cells equil nve seed rho T rc prefix\n", stderr);
        return EXIT_FAILURE;
    }
    const int cells = std::atoi(argv[1]);
    const int equil = std::atoi(argv[2]);
    const int nve = std::atoi(argv[3]);
    const auto seed = static_cast<unsigned>(std::atoi(argv[4]));
    const double rho = std::atof(argv[5]);
    const double temp = std::atof(argv[6]);
    const double rc = std::atof(argv[7]);
    const std::string prefix = argv[8];

    md::System s = md::make_fcc(cells, rho, temp, seed);
    s.cutoff = rc;
    md::NeighborList nl;
    nl.build(s);
    double pe = md::compute_forces(s, nl);

    std::puts("phase,step,time,kinetic,potential_shifted,potential_plain,total,temperature,pressure");
    for (int step = 1; step <= equil; ++step) {
        pe = md::verlet_step(s, kDt, nl);
        md::berendsen_rescale(s, temp, kDt, kTau);
        if (step % kThermoEvery == 0) print("equil", step, s, pe);
    }

    md::MsdTracker msd(s);
    md::Rdf rdf(kRdfBins, s.box / 2.0);
    std::FILE* msd_file = std::fopen((prefix + "_msd.csv").c_str(), "w");
    if (msd_file == nullptr) return EXIT_FAILURE;
    std::fputs("time,msd\n", msd_file);

    for (int step = 1; step <= nve; ++step) {
        pe = md::verlet_step(s, kDt, nl);
        if (step % kThermoEvery == 0) {
            print("nve", step, s, pe);
            msd.update(s);
            std::fprintf(msd_file, "%.4f,%.8f\n", step * kDt, msd.msd());
        }
        if (step % kRdfEvery == 0) rdf.accumulate(s);
    }
    std::fclose(msd_file);

    std::FILE* rdf_file = std::fopen((prefix + "_rdf.csv").c_str(), "w");
    if (rdf_file == nullptr) return EXIT_FAILURE;
    std::fputs("r,g\n", rdf_file);
    const auto g = rdf.g();
    for (std::size_t k = 0; k < g.size(); ++k)
        std::fprintf(rdf_file, "%.5f,%.6f\n", rdf.r(static_cast<int>(k)), g[k]);
    std::fclose(rdf_file);
    return EXIT_SUCCESS;
}
