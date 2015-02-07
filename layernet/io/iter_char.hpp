#ifndef CHAR_ITER_INL_HPP
#define CHAR_ITER_INL_HPP
#pragma once
#include <iostream>
#include <Python.h>
#include <map>
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
	class CHARIterator : public IIterator<DataBatch> {
		public:
			CHARIterator(DataProto datap) :
					datap(datap){
				mode_ = 1;
				counter = 0;
				inst_offset_ = 0;
				silent_ = 0;
				shuffle_ = datap.shuffle();
				path_img = datap.path_image();
				fileindex = datap.fileindex();
				CHECK(datap.batchsize()==1) << "batching not allowed";
			}
			virtual ~CHARIterator(void){
			}

			virtual void Init(){
				this->loadChar();
				loc_ = 0;

				out_.data = mshadow::NewTensor<cpu>(
						mshadow::Shape4(1,datap.t(),datap.batchsize(),
								datap.inputsize()),0.0);
				//	out_.data.dptr = tmp.dptr;
				out_.data.shape = mshadow::Shape4(1,datap.t(),datap.batchsize(),
						datap.inputsize());

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
				counter = 0;
			}
			virtual bool Next(void){
				if(++counter % datap.numrepeat() != 0){
					out_.data.dptr = text_[loc_].dptr;
					return true;
				}
				else {
					if(loc_ + datap.t() < text_.shape[1]) {
						out_.data.dptr = text_[loc_].dptr;
						loc_ += 1;
						return true;
					}
					else {
						return false;
					}
				}
			}
			virtual const DataBatch &Value(void) const{
				return out_;
			}

		private:
			void loadChar(){
				using namespace std;
				map<char,int> mapCI;
				char ch;
				fstream fin(path_img.c_str(),fstream::in);
				int count = 0;
				while(fin >> noskipws >> ch) {
					count++;
					mapCI[ch]++;
				}
				map<char,int>::iterator pos;
				int tmp = 0;
				for(pos = mapCI.begin(); pos != mapCI.end() ; ++pos) {
					pos->second = tmp++;
				}
				cout << mapCI.size() << endl;
				fin.close();

				text_.shape = mshadow::Shape2(count,mapCI.size());
				text_.shape.stride_ = text_.shape[0];

				// allocate continuous memory
				text_.dptr = new real_t[text_.shape.MSize()];
				text_=0.0;
				fin.open(path_img.c_str(),fstream::in);
	     count = 0;
				while(fin >> noskipws >> ch) {
					text_[count++][mapCI[ch]] = 1;
				}
				fin.close();
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
			// current location
			unsigned int loc_;
			mshadow::Tensor<cpu,2> text_;

			unsigned inst_offset_;
			// instance index
			std::vector<unsigned> inst_;
	}
	;
}
;
#endif
