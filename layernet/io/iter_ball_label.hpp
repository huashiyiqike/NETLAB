#ifndef BALL_LABEL_ITER_INL_HPP
#define BALL_LABEL_ITER_INL_HPP
#pragma once
#include <iostream>
#include <Python.h>
#include "leveldb/db.h"
#include "mshadow/tensor_container.h"
#include "data.h"
#include "../utils/io.h"
#include "../utils/global_random.h"
#include "../core/core.h"

namespace layernet {
	typedef mshadow::cpu cpu;
	typedef mshadow::gpu gpu;
	typedef mshadow::index_t index_t;
	typedef mshadow::real_t real_t;
}
;

namespace layernet {
	class BALL_LABEL_Iterator : public IIterator<DataBatch> {
		public:
			BALL_LABEL_Iterator(DataProto datap) :
					datap(datap),rnd(0){
				mode_ = 1;
				counter=0;
				inst_offset_ = 0;
				silent_ = 0;
				shuffle_ = datap.shuffle();
				path_img = datap.path_image();
				fileindex = datap.fileindex();
			}
			virtual ~BALL_LABEL_Iterator(void){
			}

			// intialize iterator loads data in
			virtual void Init(){
        loc_=0;

				out_.data = mshadow::NewTensor<cpu>(
						mshadow::Shape4(1,datap.t(),datap.batchsize(),
								datap.inputsize()),0.0);
				//	out_.data.dptr = tmp.dptr;
				out_.data.shape = mshadow::Shape4(1,datap.t(),datap.batchsize(),
 					datap.inputsize());

				out_.labels.shape = mshadow::Shape4(1,datap.t(),datap.batchsize(),
							datap.inputsize());
				out_.labels = mshadow::NewTensor<cpu>(out_.data.shape,0.0);

				out_.inst_index = NULL;
				out_.data.shape.stride_ = out_.data.shape[0];
				out_.labels.shape.stride_ = out_.data.shape[0];

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
						py.load(out_.data,fileindex,loc_,
								loc_ + datap.batchsize(),datap.pyname());
						Copy(out_.labels,out_.data);
						rnd.SampleGaussian(out_.data,0,0.1);
						out_.data+=	out_.labels;
#endif
						loc_ += datap.batchsize();
					}

					else {
						loc_ = 0;
#ifdef DEBUG
						out_.data=0.01;
#else
						py.load(out_.data,++fileindex,loc_,
								loc_ + datap.batchsize(),datap.pyname());
						Copy(out_.labels,out_.data);
						rnd.SampleGaussian(out_.data,0,0.1);
						out_.data+=	out_.labels;
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
			mshadow::Random<cpu> rnd;
			unsigned inst_offset_;
			// instance index
			std::vector<unsigned> inst_;
	};
}
;

#endif







