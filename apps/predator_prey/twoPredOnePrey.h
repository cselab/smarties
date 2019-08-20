#include <cmath>
#include <random>
#include <vector>
#include <array>
#include <functional>

#ifdef PLOT_TRAJ
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
	std::vector<double> xData3 = std::vector<double>(plotDataSize, 0);
	std::vector<double> yData3 = std::vector<double>(plotDataSize, 0);

 public:
	Window() {
    plt::figure();
    plt::figure_size(320, 320);
  }

  void update(int step, int sim, std::array<double, 2> x1, std::array<double, 2> x2, std::array<double, 2> x3)
  {
    //printf("%d %g %g %g %g\n", step, x1, y1, x2, y2); fflush(0);
    if(sim % SAVEFREQ || step % STEPFREQ) return;
    if(step>plotDataSize) step = plotDataSize;
    std::fill(xData1.data() + step, xData1.data() + plotDataSize, x1[0]);
    std::fill(yData1.data() + step, yData1.data() + plotDataSize, x1[1]);
    std::fill(xData2.data() + step, xData2.data() + plotDataSize, x2[0]);
    std::fill(yData2.data() + step, yData2.data() + plotDataSize, x2[1]);
    std::fill(xData3.data() + step, xData3.data() + plotDataSize, x3[0]);
    std::fill(yData3.data() + step, yData3.data() + plotDataSize, x3[1]);
    plt::clf();
    plt::xlim(-0.1, 1.1);
    plt::ylim(-0.1, 1.1);
    plt::plot(xData1, yData1, "r-");
    plt::plot(xData2, yData2, "m-");
    plt::plot(xData3, yData3, "b-");
    std::vector<double> X1(1,x1[0]), Y1(1,x1[1]), X2(1,x2[0]), Y2(1,x2[1]), X3(1,x3[0]), Y3(1,x3[1]);
    plt::plot(X1, Y1, "or");
    plt::plot(X2, Y2, "om");
    plt::plot(X3, Y3, "ob");
    //plt::show(false);
    char temp[32]; 
    sprintf(temp, "%05d", step);
    std::string temp2 = temp;
    plt::save("./"+std::to_string(sim)+"_"+temp2+".png");
  }
};
#endif

struct Entity
{
  //const unsigned nQuadrants; // NOTE: not used at the moment. Should we just stick to angles??
  const unsigned nStates;
  const double maxSpeed;
  std::mt19937& genA;
  bool isOver=false;

  Entity(std::mt19937& _gen, const unsigned nQ, const double vM)
    : nStates(nQ), maxSpeed(vM), genA(_gen) {}

  std::array<double, 2> p;
  double speed; 

  void reset() {
    std::uniform_real_distribution<double> distrib(0, EXTENT);
    p[0] = distrib(genA);
    p[1] = distrib(genA);
    speed = maxSpeed;
	  isOver = false;
	}

  bool is_over() {
    return isOver; // TODO add catching condition - EHH??
  }

  void advance(std::vector<double> act) {
    assert(act.size() == 2);
    speed = std::sqrt(act[0]*act[0] + act[1]*act[1]);

    if( speed > maxSpeed) { // Rescale the u and v components so that speed = maxSpeed
      p[0] += act[0] * (maxSpeed / speed) *dt;
      p[1] += act[1] * (maxSpeed / speed) *dt;
      speed = maxSpeed;
    } else {
      p[0] += act[0] *dt;
      p[1] += act[1] *dt;
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
  }

  template<typename T>
  unsigned getAngle(const T& E) const {
    double relX = E.p[0] - p[0];
    double relY = E.p[1] - p[1];
    #ifdef PERIODIC
      if(relX >  EXTENT/2) relX -= EXTENT; // WUT????
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
  const double stdNoise; // Only prey assumed to suffer from noise

  Prey(std::mt19937& _gen, const unsigned _nStates, const double vM, const double dN)
    : Entity(_gen, _nStates, vM), stdNoise(dN) {}

  template<typename T>
  std::vector<double> getState(const T& E1, const T& E2) { // wrt enemy E
    std::vector<double> state(nStates, 0);
    state[0] = p[0];
    state[1] = p[1];
    const double angEnemy1 = getAngle(E1);
    const double angEnemy2 = getAngle(E2);
    const double distEnemy1 = getDistance(E1);
    const double distEnemy2 = getDistance(E2);
    const double distEnemy = distEnemy1*distEnemy2;
    // close? low noise. moving slow? low noise
    //const double noiseAmp = stdNoise*distEnemy*speed/std::pow(maxSpeed,2);
    // or, can also think of it as ETA (Estimated Time of Arrival)
    const double ETA = distEnemy/fabs(speed);
    const double noiseAmp = stdNoise*ETA;
    std::normal_distribution<double> distrib(0, noiseAmp); // mean=0, stdDev=noiseAmp
    const double noisyAng1 = angEnemy1 + distrib(genA);
    const double noisyAng2 = angEnemy2 + distrib(genA);
    state[2] = distEnemy1*std::cos(noisyAng1); 
    state[3] = distEnemy1*std::sin(noisyAng1);
    state[4] = distEnemy2*std::cos(noisyAng2);
    state[5] = distEnemy2*std::sin(noisyAng2);
    return state;
  }

  template<typename T>
  double getReward(const T& E1, const T& E2) const {
    return getDistance(E1)*getDistance(E2);
  }

  template<typename T>
  std::vector<bool> checkTermination(const T& E1, const T& E2) {
	  const double threshold= 0.01*EXTENT;
    const double dist1 = getDistance(E1);
    const double dist2 = getDistance(E2);
  	const bool caught1 = (dist1 < threshold) ? true : false;
  	const bool caught2 = (dist2 < threshold) ? true : false;
  	const std::vector<bool> gotCaught = {caught1, caught2};

  	if(caught1 || caught2) isOver = true; 
  	return gotCaught;
  }
 
};

struct Predator: public Entity
{
  const double velPenalty;
  Predator(std::mt19937& _gen, const unsigned _nStates, const double vM, const double vP)
    : Entity(_gen, _nStates, vP*vM), velPenalty(vP) {}

  template<typename T1, typename T2>
  std::vector<double> getState(const T1& _prey, const T2& _pred) const
  { // wrt enemy (or adversary) E
    std::vector<double> state(nStates, 0);
    state[0] = p[0];
    state[1] = p[1];
    const double angPrey = getAngle(_prey); // No noisy angle for predator
    const double angPred = getAngle(_pred); // No noisy angle for predator
    const double distPrey = getDistance(_prey);
    const double distPred = getDistance(_pred);
    state[2] = distPrey*std::cos(angPrey);
    state[3] = distPrey*std::sin(angPrey);
    state[4] = distPred*std::cos(angPred);
    state[5] = distPred*std::sin(angPred);
    return state;
  }

  template<typename T1, typename T2>
  double getReward(const T1& _prey, const T2& _pred) const {
#ifdef COOPERATIVE
    const double distMult = _prey.getDistance(*this) * _prey.getDistance(_pred);
    return - distMult; // cooperative predators
#else
    return - getDistance(_prey); // competitive predators
#endif
  }
};