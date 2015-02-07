#ifndef T_BALL_ITER_INL_HPP
#define T_BALL_ITER_INL_HPP
#pragma once
#include <iostream>
#include <Python.h>
#include "leveldb/db.h"
#include "mshadow/tensor_container.h"
#include "data.h"
#include "../utils/io.h"
#include "../utils/global_random.h"


namespace layernet {
	typedef mshadow::cpu cpu;
	typedef mshadow::gpu gpu;
	typedef mshadow::index_t index_t;
	typedef mshadow::real_t real_t;
}
;

namespace layernet {
	class T_BALLIterator : public IIterator<DataBatch> {
		public:
			T_BALLIterator(DataProto datap) :
					datap(datap){
				mode_ = 1;
				counter=0;
				inst_offset_ = 0;
				silent_ = 0;
				shuffle_ = datap.shuffle();
				path_img = datap.path_image();
				fileindex = datap.fileindex();
			}
			virtual ~T_BALLIterator(void){
			}

			// intialize iterator loads data in
			virtual void Init(){
        loc_=0;
				dataread.shape = mshadow::Shape4(1,datap.t(),datap.batchsize(),
						datap.inputsize());
				dataread=mshadow::NewTensor<cpu>(dataread.shape,0.0);

				out_.data.shape = mshadow::Shape4(1,datap.t()-datap.duration(),datap.batchsize(),
							datap.inputsize() * datap.duration());
				out_.data = mshadow::NewTensor<cpu>(out_.data.shape,0.0);


				out_.inst_index = NULL;
				out_.data.shape.stride_ = out_.data.shape[0];
				if(silent_ == 0) {
					mshadow::Shape<4> s = out_.data.shape;
					printf("BALLIterator: shuffle=%d, shape=%u,%u,%u,%u\n",
							shuffle_,s[3],s[2],s[1],s[0]);
				}
			}
			virtual void BeforeFirst(void){
				this->loc_ = 0;
				first_hasdata=false;
				counter=0;
			}
			virtual bool Next(void){
				if(++counter%datap.numrepeat()!=0&&first_hasdata)
					return true;
				else{
					first_hasdata=true;
				if((int) loc_ < datap.numsample() && fileindex < 654) {
					if(loc_ < 100) {
#ifdef DEBUG
						out_.data=0.01;
#else
						py.load(dataread,fileindex,loc_,
								loc_ + datap.batchsize(),datap.pyname());
#endif
						loc_ += datap.batchsize();
#pragma omp parallel for
						for(int t=0;t<datap.t()-datap.duration();t++){
							for(int i=0;i<datap.inputsize();i++){
								out_.data[0][t][0][i]=dataread[0][t][0][i];
								out_.data[0][t][0][datap.inputsize()+i]=dataread[0][t+1][0][i];
								out_.data[0][t][0][datap.inputsize()*2+i]=dataread[0][t+2][0][i];
							}
						}

					}

					else {
						loc_ = 0;
#ifdef DEBUG
						out_.data=0.01;
#else
						py.load(dataread,++fileindex,loc_,
								loc_ + datap.batchsize(),datap.pyname());
#pragma omp parallel for
						for(int t=0;t<datap.t()-datap.duration();t++){
							for(int i=0;i<datap.inputsize();i++){
								out_.data[0][t][0][i]=dataread[0][t][0][i];
								out_.data[0][t][0][datap.inputsize()+i]=dataread[0][t+1][0][i];
								out_.data[0][t][0][datap.inputsize()*2+i]=dataread[0][t+2][0][i];
							}
						}

#endif
						loc_ += datap.batchsize();
					}
					return true;
				}
				else {
					loc_ = 0;
					++fileindex;
					if(fileindex>=654) fileindex=1;
					return false;

				}
			}
			}
			virtual const DataBatch &Value(void) const{
				return out_;
			}

		private:
			Python_helper py;
			DataProto datap;
			long fileindex;
			// silent
			int silent_;
			// path
			std::string path_img,path_label;
			// output
			DataBatch out_;
			// whether do shuffle
			bool shuffle_;
			// data mode
			int mode_;
			int counter;
			bool first_hasdata;
			// current location
			unsigned int loc_;
			mshadow::Tensor<mshadow::cpu,4>  dataread;

			unsigned inst_offset_;
			// instance index
			std::vector<unsigned> inst_;
	};
}
;

#endif


