#ifndef OP_H
#define OP_H
#pragma once

#include <cmath>
#include "mshadow/tensor.h"
#include "../compile_condition.h"
namespace layernet {
	/*! \brief operations for algorithm */
	namespace op {
		struct Exp{
			MSHADOW_XINLINE static real_t Map(real_t a) {
				return  expf(a);
			}
		};
		struct Log1exp{
			MSHADOW_XINLINE static real_t Map(real_t a) {
				return  log(1.0+expf(a));
			}
		};
		struct Log1{
			MSHADOW_XINLINE static real_t Map(real_t a) {
				return  log(1.0+a);
			}
		};
		struct Abs{
			MSHADOW_XINLINE static real_t Map(real_t a) {
				return  fabs(a);
			}
		};
		struct bound{
			MSHADOW_XINLINE static real_t Map(real_t a) {
				if(a>2) return 2;
				else if(a<-2) return -2;
				else return a;
			}
		};
		struct sigmoid {
				MSHADOW_XINLINE static real_t Map(float a){
					return 1.0 / (1.0 + exp(-a));
				}
		};
		struct integer {
				MSHADOW_XINLINE static int Map(float a){
					return int(a);
				}
		};

		struct sigmoid_grad {
				MSHADOW_XINLINE static real_t Map(real_t a){
					return a * (1.0 - a);
				}
		};

		struct mytanh {
				MSHADOW_XINLINE static real_t Map(float a){
					return 2.0 / (1.0 + exp(-a)) - 1.0;
				}
		};
		struct mytanh_grad {
				MSHADOW_XINLINE static real_t Map(float a){
					return 0.5*(1+a)*(1-a);//      2.0 / (1.0 + exp(-2 * a)) - 1.0;
				}
		};

		struct sigmoidn {
				MSHADOW_XINLINE static real_t Map(float a){
#ifdef DEBUG
					return 1.0 / (1.0 + exp(-a));
#else
					return 1.0 / (1.0 + expf(-a*3.75));
#endif
				}
		};

		struct sigmoidn_grad {
				MSHADOW_XINLINE static real_t Map(real_t a){
#ifdef DEBUG
					return a * (1.0 - a);
#else
					return 3.75*a * ( 1.0 - a );
#endif
				}
		};

		/*! \brief Rectified Linear Operation */
		struct relu {
				MSHADOW_XINLINE static real_t Map(real_t a){
					using namespace std;
					return max(a,real_t(0.0));
				}
		};
		struct relu_grad {
				MSHADOW_XINLINE static real_t Map(real_t a){
					return a > 0.0 ? 1.0 : 0.0;
				}
		};

		struct tanh {
				MSHADOW_XINLINE static real_t Map(real_t a){
					return tanhf(a);
				}
		};
		struct tanh_grad {
				MSHADOW_XINLINE static real_t Map(real_t a){
					return 1.0 - a * a;
				}
		};
		struct softplus {
				MSHADOW_XINLINE static real_t Map(real_t a){
					return logf(1 + expf(a));
				}
		};
		struct softplus_grad {
				MSHADOW_XINLINE static real_t Map(real_t a){
					return 1.0f / (1.0f + expf(-a));
				}
		};
		struct bnll {
				MSHADOW_XINLINE static real_t Map(real_t a){
					return a > 0.0 ?
							a + logf(1.0 + expf(-a)) : logf(1.0 + expf(a));
				}
		};
		struct bnll_grad {
				MSHADOW_XINLINE static real_t Map(real_t a){
					real_t expval = a > 50.0 ? 50.0 : a; // kBNLL_THRESHOLD = 50.0f
					expval = expf(-expval);
					return 1.0 / (1.0 + expval);
				}
		};

		struct square {
				MSHADOW_XINLINE static real_t Map(real_t a){
					return a * a;
				}
		};
	}
	;

}
;

namespace layernet {
	namespace op {
		struct samples{
			MSHADOW_XINLINE static real_t Map(real_t a,real_t b) {
				return  a>b?1.0:0.0;
			}
		};
		struct fmode {
				MSHADOW_XINLINE static real_t Map(real_t a,real_t b){
					return fmod(a,b);
				}
		};
		/*! \brief used for generate Bernoulli mask */
		struct threshold {
				MSHADOW_XINLINE static real_t Map(real_t a,real_t b){
					return a < b ? 1.0 : 0.0;
				}
		};

		/*! \brief used for generate element of power */
		struct power {
				MSHADOW_XINLINE static real_t Map(real_t a,real_t b){
					return powf(a,b);
				}
		};
	}
	;

}
;

#endif
