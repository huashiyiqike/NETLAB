/*
 * iter_mult.hpp
 *
 *  Created on: 2014-9-26
 *      Author: lq
 */

#ifndef MULT_ITER_INL_HPP
#define MULT_ITER_INL_HPP
#pragma once
#include <iostream>
#include "leveldb/db.h"
#include "mshadow/tensor_container.h"
#include "data.h"
#include "../utils/io.h"
#include "../utils/global_random.h"
#include <glog/logging.h>


namespace layernet {
	typedef mshadow::cpu cpu;
	typedef mshadow::gpu gpu;
	typedef mshadow::index_t index_t;
	typedef mshadow::real_t real_t;
}
;

namespace layernet {
	class MULTIterator : public IIterator<DataBatch> {
		public:
			MULTIterator(DataProto datap) :
					datap(datap), rnd(0){
				mode_ = 1;
				inst_offset_ = 0;
				silent_ = 0;
				loc_ = 0;
				shuffle_ = datap.shuffle();
				path_img = datap.path_image();
				fileindex = datap.fileindex();
				CHECK(datap.numsample() >= datap.batchsize());
			}
			virtual ~MULTIterator(void){
				//	if(img_.dptr != NULL) delete[] img_.dptr;

			}

			// intialize iterator loads data in
			virtual void Init(){
				out_.data = mshadow::NewTensor<cpu>(
						mshadow::Shape4(1,datap.t(),datap.batchsize(),
								datap.inputsize()),0.0);
				out_.labels = mshadow::NewTensor<cpu>(
						mshadow::Shape4(1,datap.t(),datap.batchsize(),1),0.0);
				out_.lengthlist = mshadow::NewTensor<cpu>(
						mshadow::Shape1(datap.batchsize()),0.0);

				out_.inst_index = NULL;
				out_.data.shape.stride_ = out_.data.shape[0];
				out_.labels.shape.stride_ = out_.labels.shape[0];
				if(silent_ == 0) {
					mshadow::Shape<4> s = out_.data.shape;
					printf("MULTIterator: shuffle=%d, shape=%u,%u,%u,%u\n",
							shuffle_,s[3],s[2],s[1],s[0]);
				}
			}
			virtual void BeforeFirst(void){
				this->loc_ = 0;
			}

			virtual bool Next(void){
				if((int) loc_ < datap.numsample()) {
					generateData();
					loc_ += datap.batchsize();
					return true;
				}
				else {
					loc_ = 0;
					return false;
				}
			}
			virtual const DataBatch &Value(void) const{
				return out_;
			}
		private:
			void generateData(){
				out_.labels = 0.0;
				int T = datap.t() ;/// 1.1;
#ifdef DEBUG
				out_.data=1.0;
				out_.lengthlist[0]=9;
				out_.labels=1.0;
#else
				rnd.SampleUniform(out_.data,0.0,1.0);
				for(int i = 0 ; i < datap.batchsize() ; i++) {
					for(int t = 0 ; t < datap.t() ; t++) {
						out_.data[0][t][i][1] = 0;
					}
				}

				for(int i = 0 ; i < datap.batchsize() ; i++) {

					int length = T;//utils::NextDouble() * T / 10;
					out_.lengthlist[i] = length ;
					//		LOG(INFO)<<i<<" "<<length<<std::endl;
					//	mask[length][i][0]=1;
					int tmp1,tmp2;
					if(T > 10) {
						tmp1 = utils::NextDouble() * 10;
					}
					else {
						tmp1 = utils::NextDouble() * datap.t() / 2;
					}
					out_.data[0][tmp1][i][1] = 1;

					tmp2 = tmp1;
					while(tmp2 == tmp1) {
						tmp2 = utils::NextDouble() * datap.t() / 2;
					}
					out_.data[0][tmp2][i][1] = 1;
					if(tmp2 == 0 || tmp1 == 0) {
						out_.data[0][0][i][1] = 0;
					}
					else {
						out_.data[0][0][i][1] = -1;
					}
					out_.data[0][length - 2][i][1] = -1;
					// it's possible for label to be 0, necessitate lengthlist
					out_.labels[0][length - 1][i][0] = (out_.data[0][tmp1][i][0]
							+ out_.data[0][tmp2][i][0]) / 2;

					if(datap.spy()) {
						for(size_t t = 0 ; t < out_.data.shape[2] ; t++) {
							for(unsigned int i = 0 ; i < out_.data.shape[1] ;
									i++)
								LOG(INFO)<<"T "<<t<<"sampele "<<i<<" "<<out_.data[0][t][i][0]<<" "<<out_.data[0][t][i][1]<<" length "<<out_.lengthlist[i];

								LOG(INFO)<<out_.labels[0][t][0][0];
							}
						}
					}
#endif
				}
				private:
				DataProto datap;
				long fileindex;
				// silent
				int silent_;
				// path
				std::string path_img,path_label;
				mshadow::Random<cpu> rnd;
				// output
				DataBatch out_;
				// whether do shuffle
				bool shuffle_;
				// data mode
				int mode_;
				// current location
				unsigned int loc_;

				unsigned inst_offset_;
				// instance index
				std::vector<unsigned> inst_;
			}
			;

		}
		;

#endif
