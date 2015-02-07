#ifndef CLOTHES_ITER_INL_HPP
#define CLOTHES_ITER_INL_HPP
#pragma once
#include <iostream>
#include <fstream>
#include <Python.h>
#include <map>
#include "lmdb.h"
#include "mshadow/tensor_container.h"
#include "data.h"
#include "../../layernet.pb.h"
#include "../utils/io.h"
#include "../utils/global_random.h"
using namespace std;
namespace layernet {
typedef mshadow::cpu cpu;
typedef mshadow::gpu gpu;
typedef mshadow::index_t index_t;
typedef mshadow::real_t real_t;
}
;

namespace layernet {
class CLOTHESIterator: public IIterator<DataBatch> {
public:
	CLOTHESIterator(DataProto datap) :
			datap(datap) {
		mode_ = 1;
		counter = 0;
		inst_offset_ = 0;
		silent_ = 0;
		shuffle_ = datap.shuffle();
		path_img = datap.path_image();
		fileindex = datap.fileindex();
		CHECK(datap.batchsize() == 1) << "batching not allowed";
	}
	virtual ~CLOTHESIterator(void) {
	}

	// intialize iterator loads data in
	virtual void Init() {
		this->loadText();
		loc_ = 0;
		CHECK(datap.t() == 1) << " t = 1! ";
		out_.data.shape = mshadow::Shape4(1, datap.t(), datap.batchsize(),
				datap.inputsize());
		out_.data = mshadow::NewTensor<cpu>(out_.data.shape, 0.0);
		out_.data.shape.stride_ = out_.data.shape[0];
		if (silent_ == 0) {
			mshadow::Shape<4> s = out_.data.shape;
			printf("CLOTHESIterator: shuffle=%d, shape=%u,%u,%u,%u\n", shuffle_,
					s[3], s[2], s[1], s[0]);
		}
	}
	virtual void BeforeFirst(void) {
		this->loc_ = 0;
		counter = 0;
	}
	virtual bool Next(void) {
		if (++counter % datap.numrepeat() != 0) {
			out_.data.dptr = high_[loc_].dptr;
			// use lengthlist[0] as length in nnet-inl
			return true;
		} else {

			for (int i = 0; i < datap.batchsize(); i++) {
				mdb_key.mv_size = highvec[loc_].size(); //sizeof(tmpstr);
				mdb_key.mv_data = reinterpret_cast<void*>(&highvec[loc_][0]);
				CHECK_EQ(mdb_txn_begin(mdb_env, NULL, MDB_RDONLY, &mdb_txn),
						MDB_SUCCESS) << "mdb_txn_begin failed";
				Datum datum;
				datum.ParseFromArray(mdb_data.mv_data, mdb_data.mv_size);
				CHECK_EQ(mdb_get(mdb_txn, mdb_dbi, &mdb_key, &mdb_data),
						MDB_SUCCESS);
				printf("key: %p %.*s, data: %p %.*s\n", mdb_key.mv_data,
						(int) mdb_key.mv_size, (char *) mdb_key.mv_data,
						mdb_data.mv_data, (int) mdb_data.mv_size,
						(char *) mdb_data.mv_data);

				//Copy(out_.data[0][0][i],mdb_data.mv_data);

				return true;
			}
		}
	}
	virtual const DataBatch &Value(void) const {
		return out_;
	}

private:
	void loadText() {
		using namespace std;
		infile.open(path_img.c_str(), ifstream::in);
		//	FILE* file = fopen(path_img.c_str(),"r");
		high_.shape = mshadow::Shape4(datap.numsample(), 1, datap.batchsize(),
				datap.inputsize());
		high_ = mshadow::NewTensor<cpu>(high_.shape, 0.0);
		low_.shape = high_.shape;
		low_ = mshadow::NewTensor<cpu>(low_.shape, 0.0);

		LOG(INFO) << "Opening .." << datap.path_image();
		CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS)
				<< "mdb_env_create failed";
		CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS) // 1TB
		<< "mdb_env_set_mapsize failed";
		CHECK_EQ(mdb_env_open(mdb_env, datap.path_image().c_str(), 0, 0664),
				MDB_SUCCESS) << "mdb_env_open failed";
		CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
				<< "mdb_txn_begin failed";
		CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
				<< "mdb_open failed. Does the lmdb already exist? ";
		int labeltmp;
		for (int count = 0; count < datap.numsample(); count++) {
			infile >> highvec[count] >> lowvec[count] >> labeltmp;
		}

	}
private:
	Python_helper py;
	DataProto datap;
	long fileindex;
// silent
	int silent_;
// path
	std::string path_img, path_label;
// output
	DataBatch out_;
// whether do shuffle
	bool shuffle_;
// data mode
	int mode_;
	int counter;
// current location
	int loc_;
	fstream infile;
	mshadow::Tensor<cpu, 4> high_;
	mshadow::Tensor<cpu, 4> low_;
	vector<string> highvec;
	vector<string> lowvec;
	unsigned inst_offset_;
// instance index
	std::vector<unsigned> inst_;
	MDB_env *mdb_env;
	MDB_dbi mdb_dbi;
	MDB_val mdb_key, mdb_data;
	MDB_txn *mdb_txn;
}
;
}
;
#endif

