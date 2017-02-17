#include "host/wos_host.h"
#include "wos_native.cuh"
#include "wos_thrust.cuh"

template <typename T>
T wosWrapper(Timers &timers, Parameters &p) {
  switch (p.wos.simulation) {
  case (nativeWos):
    return wosNative<T>(timers, p);
  case (thrustWos):
    return wosThrust<T>(timers, p);
  case (hostWos):
    return (T)wosHost(timers, p);
  }
  return 0;
}
