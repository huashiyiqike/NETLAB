#ifndef T_TEXT_ITER_INL_HPP
#define T_TEXT_ITER_INL_HPP
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
	class T_TEXTIterator : public IIterator<DataBatch> {
		public:
			T_TEXTIterator(DataProto datap) :
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
			virtual ~T_TEXTIterator(void){
			}

			// intialize iterator loads data in
			virtual void Init(){
				this->loadText();
				loc_ = 0;

				CHECK(datap.batchsize()==1) << "Do not support batch now";

				out_.data.shape = mshadow::Shape4(1,
						datap.t() - datap.duration() ,datap.batchsize(),
						datap.inputsize() * datap.duration());
				out_.data = mshadow::NewTensor<cpu>(out_.data.shape,0.0);

				dataread.shape = mshadow::Shape4(1,
						datap.t() - datap.duration() ,1,datap.inputsize()* datap.duration());
				dataread = mshadow::NewTensor<cpu>(dataread.shape,0.0);

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
				if(++counter % datap.numrepeat() != 0) {
					out_.lengthlist.dptr=lengthlist_[loc_].dptr;

					for(int t = 0 ; t < lengthlist_[loc_][0] ; t++) {
						for(int i = 0 ; i < datap.inputsize() ; i++) {
							dataread[0][t][0][i] = text_[loc_][t][0][i];
							dataread[0][t][0][datap.inputsize() + i] =
									text_[loc_][t + 1][0][i];
							dataread[0][t][0][datap.inputsize() * 2 + i] =
									text_[loc_][t + 2][0][i];
						}
					}

					out_.data.dptr = dataread[0].dptr;

					return true;
				}
				else {
					if(loc_ + 1 < text_.shape[3]) {
						out_.lengthlist.dptr=lengthlist_[loc_].dptr;
						for(int t = 0 ; t <lengthlist_[loc_][0]; t++) {
							for(int i = 0 ; i < datap.inputsize() ; i++) {
								dataread[0][t][0][i] = text_[loc_][t][0][i];
								dataread[0][t][0][datap.inputsize() + i] =
										text_[loc_][t + 1][0][i];
								dataread[0][t][0][datap.inputsize() * 2 + i] =
										text_[loc_][t + 2][0][i];
							}
						}
						out_.data.dptr = dataread[0].dptr;
						loc_ += 1;
						out_.lengthlist.dptr=lengthlist_[loc_].dptr;;
						return true;
					}
					else {
						this->loc_ = 0;
						counter = 0;
						return false;
					}
				}
			}
			virtual const DataBatch &Value(void) const{
				return out_;
			}

		private:
			void loadText(){
				using namespace std;
				ifstream infile(path_img.c_str(),ifstream::in);
				//	FILE* file = fopen(path_img.c_str(),"r");
				int total_count,lines,words,max_length;
				infile >> total_count >> max_length >> words;
				cout << "%%%%%%  total count:" << total_count << " max_length:"
						<< max_length << " total words:" << words << endl;
				text_ = mshadow::NewTensor<cpu>(mshadow::Shape4(total_count,max_length,1,words),0.0);
				out_.lengthlist = mshadow::NewTensor<cpu>(
						mshadow::Shape1(total_count),0.0);
				lengthlist_= mshadow::NewTensor<cpu>(
						mshadow::Shape2(total_count,1),0.0);
				//total_count for one text now
				for(int count = 0 ; count < total_count ; count++) {
					infile >> lines;
					//	cout << count << " " << lines << endl;
					lengthlist_[count][0] = lines - datap.duration() + 1;

					for(int i = 0 ; i < lines ; i++) {
						for(int j = 0 ; j < words ; j++) {
							try {
								infile >> text_[count][i][0][j];
							}
							catch(const std::ifstream::failure& e) {
								std::cout << "Exception!!!! " << e.what();
							}
						}
					}
				}
				infile.close();
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
			mshadow::Tensor<cpu,4> dataread;
			mshadow::Tensor<cpu,4> text_;
			mshadow::Tensor<cpu,2> lengthlist_;
			unsigned inst_offset_;
			// instance index
			std::vector<unsigned> inst_;
	}
	;
}
;
#endif
