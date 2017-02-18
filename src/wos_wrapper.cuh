#include "host/wos_host.h"
#include "wos_native.cuh"
#include "wos_thrust.cuh"

float wosWrapper(Timers &timers, Parameters &p) {
  switch (p.wos.simulation) {
  case (nativeWos):
    return wosNative(timers, p);
  case (thrustWos):
    return wosThrust(timers, p);
  case (hostWos):
    return wosHost(timers, p);
  }
  return 0;
}
