#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

// CPU version
#include "nnet-inl.hpp"

namespace layernet {
    INetTrainer* CreateNetCPU( Task task, int net_type){
        return CreateNet_<cpu>(task, net_type );
    }
    #if MSHADOW_USE_CUDA == 0
    INetTrainer* CreateNetGPU( int net_type ){
        utils::Error("CUDA is not supported");
        return NULL;
    }    
    #endif
};
