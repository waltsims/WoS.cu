#include "helper_cuda.h"
#include "host/wos_host.h"
#include "wos_native.cuh"
#include "wos_thrust.cuh"

#include <limits>
#include <math_functions.h>

float wosWrapper(Timers &timers, Parameters &p) {
  switch (p.simulation) {
  case (nativeWos):
    return wosNative(timers, p);
  case (thrustWos):
    return wosThrust(timers, p);
  case (hostWos):
    return wosHost(timers, p);
  }
  return 0;
}
