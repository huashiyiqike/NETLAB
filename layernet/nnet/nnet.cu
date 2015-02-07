#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
// GPU version
#include "nnet-inl.hpp"

namespace layernet {
    INetTrainer* CreateNetGPU(Task task, int net_type ){
        return CreateNet_<gpu>(task, net_type );
    }
};
