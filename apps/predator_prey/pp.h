#include <cmath>
#include <random>
#include <vector>
#include <array>
#include <functional>

#define EXTENT 1.0
#define SAVEFREQ 1000
#define STEPFREQ 1
//#define PERIODIC

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

class Window
{
 private:
	static constexpr int plotDataSize = 500;
	std::vector<double> xData1 = std::vector<double>(plotDataSize, 0);
	std::vector<double> yData1 = std::vector<double>(plotDataSize, 0);
	std::vector<double> xData2 = std::vector<double>(plotDataSize, 0);
	std::vector<double> yData2 = std::vector<double>(plotDataSize, 0);

 public:
	Window() {
    plt::figure();
    plt::figure_size(320, 320);
  }

  void update(int step, int sim, double x1, double y1,
              double x2, double y2)
  {
    //printf("%d %g %g %g %g\n", step, x1, y1, x2, y2); fflush(0);
    if(sim % SAVEFREQ || step % STEPFREQ) return;
    if(step>plotDataSize) step = plotDataSize;
    std::fill(xData1.data() + step, xData1.data() + plotDataSize, x1);
    std::fill(yData1.data() + step, yData1.data() + plotDataSize, y1);
    std::fill(xData2.data() + step, xData2.data() + plotDataSize, x2);
    std::fill(yData2.data() + step, yData2.data() + plotDataSize, y2);
    plt::clf();
    plt::xlim(-0.1, 1.1);
    plt::ylim(-0.1, 1.1);
    plt::plot(xData1, yData1, "r-");
    plt::plot(xData2, yData2, "b-");
    std::vector<double> X1(1,x1), Y1(1,y1), X2(1,x2), Y2(1,y2);
    plt::plot(X1, Y1, "or");
    plt::plot(X2, Y2, "ob");
    //plt::show(false);
    plt::save("./"+std::to_string(sim)+"_"+std::to_string(step)+".png");
  }
};

struct Entity
{
  const unsigned nQuadrants;
  const double velMagnitude;
  Entity(const unsigned nQ, const double vM)
    : nQuadrants(nQ), velMagnitude(vM) {}

  std::array<double, 2> p;
  double actScal;

  void reset(std::mt19937& gen) {
    std::uniform_real_distribution<double> dist(0, EXTENT);
    p[0] = dist(gen);
    p[1] = dist(gen);
    actScal = velMagnitude; // so that prey overwrites background
	}

  bool is_over() {
    return false; // TODO add catching condition
  }

  int advance(std::vector<double> act) {
    assert(act.size() == 2);
    actScal = std::sqrt(act[0]*act[0] + act[1]*act[1]);
    if( actScal > velMagnitude) {
      p[0] += act[0] * velMagnitude / actScal;
      p[1] += act[1] * velMagnitude / actScal;
      actScal = velMagnitude;
    } else {
      p[0] += act[0];
      p[1] += act[1];
    }
    #ifdef PERIODIC
      if (p[0] >= EXTENT) p[0] -= EXTENT;
      if (p[0] <  0)      p[0] += EXTENT;
      if (p[1] >= EXTENT) p[1] -= EXTENT;
      if (p[1] <  0)      p[1] += EXTENT;
    #else
      if (p[0] > EXTENT) p[0] = EXTENT;
      if (p[0] < 0)      p[0] = 0;
      if (p[1] > EXTENT) p[1] = EXTENT;
      if (p[1] < 0)      p[1] = 0;
    #endif
    return is_over();
  }

  template<typename T>
  unsigned getAngle(const T& E) const {
    double relX = E.p[0] - p[0];
    double relY = E.p[1] - p[1];
    #ifdef PERIODIC
      if(relX >  EXTENT/2) relX -= EXTENT;
      if(relX < -EXTENT/2) relX += EXTENT;
      if(relY >  EXTENT/2) relY -= EXTENT;
      if(relY < -EXTENT/2) relY += EXTENT;
    #endif
    return std::atan2(relY, relX);
  }

  template<typename T>
  double getDistance(const T& E) const {
    double relX = E.p[0] - p[0];
    double relY = E.p[1] - p[1];
    #ifdef PERIODIC
      if(relX >  EXTENT/2) relX -= EXTENT;
      if(relX < -EXTENT/2) relX += EXTENT;
      if(relY >  EXTENT/2) relY -= EXTENT;
      if(relY < -EXTENT/2) relY += EXTENT;
    #endif
    return std::sqrt(relX*relX + relY*relY);
  }
};

struct Prey: public Entity
{
  const double stdNoise;

  Prey(const unsigned nQ, const double vM, const double dN)
    : Entity(nQ, vM), stdNoise(dN) {}

  template<typename T>
  std::vector<double> getState(const T& E, std::mt19937& gen) {
    std::vector<double> state(4, 0);
    state[0] = p[0];
    state[1] = p[1];
    const double angEnemy = getAngle(E);
    const double disEnemy = getDistance(E);
    // close? low noise. moving slow? low noise
    const double noiseAmp = stdNoise*disEnemy*actScal/std::pow(velMagnitude,2);
    std::normal_distribution<double> dist(0, noiseAmp);
    const double noiseAng = angEnemy + dist(gen);
    state[2] = std::cos(noiseAng);
    state[3] = std::sin(noiseAng);
    return state;
  }

  template<typename T>
  double getReward(const T& E) const {
    return getDistance(E);
  }
};

struct Predator: public Entity
{
  const double velPenalty;
  Predator(const unsigned nQ, const double vM, const double vP)
    : Entity(nQ, vP*vM), velPenalty(vP) {}

  template<typename T>
  std::vector<double> getState(const T& E) const {
    std::vector<double> state(4, 0);
    state[0] = p[0];
    state[1] = p[1];
    const double angEnemy = getAngle(E);
    state[2] = std::cos(angEnemy);
    state[3] = std::sin(angEnemy);
    return state;
  }

  template<typename T>
  double getReward(const T& E) const {
    return - getDistance(E);
  }
};
